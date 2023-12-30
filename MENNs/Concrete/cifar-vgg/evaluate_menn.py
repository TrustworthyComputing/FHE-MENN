import argparse
import pathlib

import numpy as np
import torch
from concrete.fhe.compilation.configuration import Configuration
from concrete.fhe.compilation.artifacts import DebugArtifacts
from models import cnv_2w2a
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from trainer import accuracy
import time

import scipy

from concrete.ml.torch.compile import compile_brevitas_qat_model
def measure_execution_time(func):
    """Run a function and return execution time and outputs.

    Usage:
        def f(x, y):
            return x + y
        output, execution_time = measure_execution_time(f)(x,y)
    """

    def wrapper(*args, **kwargs):
        # Get the current time
        start = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # Get the current time again
        end = time.time()

        # Calculate the execution time
        execution_time = end - start

        # Return the result and the execution time
        return result, execution_time

    return wrapper


def cml_inference(quantized_numpy_module, x_numpy):
    predictions = np.zeros(shape=(x_numpy.shape[0], 10))
    for idx, x in enumerate(x_numpy):
        x_q = np.expand_dims(x, 0)
        predictions[idx] = quantized_numpy_module.forward(x_q, fhe="simulate")
    return predictions


def evaluate(torch_model, cml_model, ei):
    # Import the CIFAR data (following bnn_pynq_train.py)

    transform_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]  # Normalizes data between -1 and +1s
    )

    builder = CIFAR10

    test_set = builder(root=".datasets/", train=False, download=True, transform=transform_to_tensor)

    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=1)

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device in use:", device)
    
    # Model to device
    for i in range(1):
        print(f"Exit {ei}", flush=True)
        torch_model.setexit(ei)
        torch_model = torch_model.to(device)
        top1_torch = []
        top5_torch = []

        top1_cml = []
        top5_cml = []

        entropies = []
        preds = []
        targets = []
        corrects = []

        entropies_cml = []
        preds_cml = []
        corrects_cml = []

        for _, data in enumerate(test_loader):
            print(".", end='')
            (input, target) = data

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Compute torch output
            output = torch_model(input)

            numpy_input = input.detach().cpu().numpy()

            # Compute Concrete ML output
            y_cml_simulated = cml_inference(cml_model, numpy_input)

            # y_cml_simulated to torch to device
            y_cml_simulated = torch.tensor(y_cml_simulated).to(device)

            # Compute torch loss
            pred = output.data.argmax(1, keepdim=True)
            entropy = scipy.stats.entropy(scipy.special.softmax(output.detach().numpy(),axis=1),axis=1)
            ans_correct = pred.eq(target.data.view_as(pred))
            
            correct = pred.eq(target.data.view_as(pred)).sum()
            prec1 = 100.0 * correct.float() / input.size(0)

            _, prec5 = accuracy(output, target, topk=(1, 5))

            top1_torch.append(prec1.item())
            top5_torch.append(prec5.item())
        
            entropies.append(entropy)
            preds.append(pred.detach().numpy())
            targets.append(target.detach().numpy())
            corrects.append(ans_correct.detach().numpy())

            # Compute Concrete ML loss
            pred = y_cml_simulated.data.argmax(1, keepdim=True)
            entropy = scipy.stats.entropy(scipy.special.softmax(y_cml_simulated.detach().numpy(),axis=1),axis=1)
            ans_correct = pred.eq(target.data.view_as(pred))
            correct = pred.eq(target.data.view_as(pred)).sum()
            prec1 = 100.0 * correct.float() / input.size(0)

            _, prec5 = accuracy(y_cml_simulated, target, topk=(1, 5))

            top1_cml.append(prec1.item())
            top5_cml.append(prec5.item())
 
            entropies_cml.append(entropy)
            preds_cml.append(pred.detach().numpy())
            corrects_cml.append(ans_correct.detach().numpy())

        print("Torch accuracy top1:", np.mean(top1_torch))
        print("Concrete ML accuracy top1:", np.mean(top1_cml))

        print("Torch accuracy top5:", np.mean(top5_torch))
        print("Concrete ML accuracy top5:", np.mean(top5_cml))
        np.savez_compressed(f"npy/entropies{ei}.npz", entropies=np.array(entropies_cml), preds=np.array(preds_cml), \
                correct=np.array(corrects_cml), target=np.array(targets))

def main(modelloc, rounding_threshold_bits_list):
    model = cnv_2w2a(False)
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # Find relative path to this file
    dir_path = pathlib.Path(__file__).parent.absolute()

    # Load checkpoint
    checkpoint = torch.load(
        dir_path / modelloc, #"../checkpoints/best.tar", #"./experiments/CNV_2W2A_2W2A_20230807_150413/checkpoints/best.tar",
        map_location=device,
    )

    # Load weights
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Get some random data with the right shape
    transform_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]  # Normalizes data between -1 and +1s
    )
    builder = CIFAR10
    test_set = builder(root=".datasets/", train=False, download=True, transform=transform_to_tensor)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=1)

    # Eval mode
    for ei in [1,0,2,-10,-11]:
        model.setexit(ei)
        model.eval()

        # Compile with Concrete ML using the FHE simulation mode
        cfg = Configuration(
            dump_artifacts_on_unexpected_failures=False,
            enable_unsafe_features=True,  # This is for our tests only, never use that in prod
            verbose=True,
            show_optimizer=True,
        )
        x = torch.randn(1000, 3, 32, 32)
        for rounding_threshold_bits in rounding_threshold_bits_list:
            aft = DebugArtifacts(output_directory=f"./artifacts/{rounding_threshold_bits}-{ei}")
            print(f"Testing network with {rounding_threshold_bits} rounding bits")

            quantized_numpy_module = compile_brevitas_qat_model(
                model,  # our torch model
                x,  # a representative input-set to be used for both quantization and compilation
                n_bits={"model_inputs": 8, "model_outputs": 8},
                configuration=cfg,
                artifacts = aft,
                rounding_threshold_bits=rounding_threshold_bits,
                verbose = True, 
            )
            aft.export() 
            # Print max bit-width in the circuit
            print(
                "Max bit-width in the circuit: ",
                quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width(),
            )

            # Evaluate torch and Concrete ML model
            evaluate(model, quantized_numpy_module, ei)
            # Key generation
            print("Creation of the private and evaluation keys.")
            _, keygen_execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.keygen)(force=False)
            print(f"Keygen took {keygen_execution_time} seconds")

            timings = []
            
            x, labels = next(iter(test_loader))
            x_numpy = x.numpy()

            for imgi in range(3):
                # Iterate through the NUM_SAMPLES
                # Take one example
                test_x_numpy = x_numpy[imgi : imgi + 1]
                print(test_x_numpy)

                # Quantize the input
                q_x_numpy, quantization_execution_time = measure_execution_time(
                    quantized_numpy_module.quantize_input)(test_x_numpy)

                print(f"Quantization of a single input (image) took {quantization_execution_time} seconds")
                print(f"Size of CLEAR input is {q_x_numpy.nbytes} bytes\n")

                encrypted_q_x_numpy, encryption_execution_time = measure_execution_time(
                quantized_numpy_module.fhe_circuit.encrypt)(q_x_numpy)
                print(f"Encryption of a single input (image) took {encryption_execution_time} seconds\n")
                print("Running FHE inference")
                fhe_output, fhe_execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.run)(
                encrypted_q_x_numpy)
                print(f"FHE inference over a single image took {fhe_execution_time} seconds")
                timings.append(fhe_execution_time)
            print(f"Timing for exit {ei}: {np.average(np.array(timings))} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounding", nargs="+", type=int, default=[None])
    parser.add_argument("--model", type=str, default="checkpoints/best.tar")
    rounding_threshold_bits_list = parser.parse_args().rounding
    main(parser.parse_args().model, rounding_threshold_bits_list)
