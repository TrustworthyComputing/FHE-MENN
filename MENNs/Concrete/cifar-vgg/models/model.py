# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Original file can be found at https://github.com/Xilinx/brevitas/blob/8c3d9de0113528cf6693c6474a13d802a66682c6/src/brevitas_examples/bnn_pynq/models/CNV.py

import torch
from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from torch.nn import AvgPool2d, BatchNorm1d, BatchNorm2d, Module, ModuleList
from os.path import exists

from .common import CommonActQuant, CommonWeightQuant
from .tensor_norm import TensorNorm

CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
EXIT0_FC_FEATURES = [(576, 256), (256, 256)]
EXIT1_FC_FEATURES = [(512, 392), (392, 392)]
EXIT2_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3
EXITS = [0,2,4,6]


class CNV(Module):
    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch, conv_stop=6, lin_stop=2):
        super(CNV, self).__init__()
        self.time_enable = False
        self.seg0, self.exit0, self.pool0 = (ModuleList(), ModuleList(), ModuleList())
        self.seg1, self.exit1, self.pool1 = (ModuleList(), ModuleList(), ModuleList())
        self.seg2, self.exit2 = (ModuleList(), ModuleList())
        self.myexit = 2
        
        self.seg0.append(
            QuantIdentity(  # for Q1.7 input format
                act_quant=CommonActQuant,
                return_quant_tensor=True,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            )
        )
        
        # Segment 0
        ei = 0
        #start and finish
        eis, eif = (EXITS[ei], EXITS[ei+1])
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL[eis:eif]:
            self.seg0.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            in_ch = out_ch
            self.seg0.append(BatchNorm2d(in_ch, eps=1e-4))
            self.seg0.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            if is_pool_enabled:
                self.seg0.append(AvgPool2d(kernel_size=2))
                self.seg0.append(
                    QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
                )

         
        # Segment 1
        ei = 1
        #start and finish
        eis, eif = (EXITS[ei], EXITS[ei+1])
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL[eis:eif]:
            self.seg1.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            in_ch = out_ch
            self.seg1.append(BatchNorm2d(in_ch, eps=1e-4))
            self.seg1.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            if is_pool_enabled:
                self.seg1.append(AvgPool2d(kernel_size=2))
                self.seg1.append(
                    QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
                )
        
        # Segment 2
        ei = 2
        #start and finish
        eis, eif = (EXITS[ei], EXITS[ei+1])
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL[eis:eif]:
            self.seg2.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            in_ch = out_ch
            self.seg2.append(BatchNorm2d(in_ch, eps=1e-4))
            self.seg2.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            if is_pool_enabled:
                self.seg2.append(AvgPool2d(kernel_size=2))
                self.seg2.append(
                    QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
                )

        # Exit 0
        self.pool0.append(AvgPool2d(kernel_size=4))
        self.pool0.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
        for in_features, out_features in EXIT0_FC_FEATURES:
            self.exit0.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            self.exit0.append(BatchNorm1d(out_features, eps=1e-4))
            self.exit0.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )

        self.exit0.append(
            QuantLinear(
                in_features=EXIT0_FC_FEATURES[-1][-1],
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width,
            )
        )
        self.exit0.append(TensorNorm())

        # Exit 1
        self.pool1.append(AvgPool2d(kernel_size=2))
        self.pool1.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
        for in_features, out_features in EXIT1_FC_FEATURES:
            self.exit1.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            self.exit1.append(BatchNorm1d(out_features, eps=1e-4))
            self.exit1.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )

        self.exit1.append(
            QuantLinear(
                in_features=EXIT1_FC_FEATURES[-1][-1],
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width,
            )
        )
        self.exit1.append(TensorNorm())

        # Exit 2
        for in_features, out_features in EXIT2_FC_FEATURES:
            self.exit2.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            self.exit2.append(BatchNorm1d(out_features, eps=1e-4))
            self.exit2.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )

        self.exit2.append(
            QuantLinear(
                in_features=LAST_FC_IN_FEATURES,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width,
            )
        )
        self.exit2.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.seg0:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.seg1:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.seg2:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)

        for mod in self.exit2:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        for mod in self.seg0:
            x = mod(x)
        if self.myexit == -10:
            return torch.flatten(x,1)[:10]
        #exit 0
        if self.myexit == 0:
            for mod in self.pool0:
                x = mod(x)
            x = torch.flatten(x,1)
            for mod in self.exit0:
                x = mod(x)
            return x
        
        for mod in self.seg1:
            x = mod(x) 
        if self.myexit == -11:
            return torch.flatten(x,1)[:10]
        #Exit 1
        if self.myexit == 1:
            for mod in self.pool1:
                x = mod(x)
            x = torch.flatten(x,1)
            for mod in self.exit1:
                x = mod(x)
            return x
        for mod in self.seg2:
            x = mod(x)
        #else final exit
        x = torch.flatten(x,1)
        for mod in self.exit2:
            x = mod(x)
        return x
    
    def load_exits(self, basepath="modelexits/", fpload= True):
        for ei in range(self.myexit):
            segfile= f"qseg{ei}.pt"
            exitfile= f"qexit{ei}.pt"
            #if last_exitfp and ei == self.myexit:
            #    self.load_fpexit(ei,  basepath)

            #if fpload or not exists(basepath + segfile) or not exists(basepath + exitfile) or qmodel is None:
            #if not exists(basepath + segfile) or not exists(basepath + exitfile):
            print(f"Exit {ei}: Quantized denied, loading floating point")
            print(f"{fpload}, {basepath} {segfile}")
            print(f"{exists(basepath + segfile)} {exitfile} {exists(basepath + exitfile)}")
            self.load_fpexit(ei,  basepath)

    def load_fpexit(self, ei, basepath="modelexits/", saveall=True):
        segfile= f"seg{ei}.pt"
        exitfile= f"exit{ei}.pt"

        if not exists(basepath + segfile) or not exists(basepath+exitfile):
            print(f"Exit {ei}: No FP Exists")
        elif ei == 0:
            self.seg0.load_state_dict(torch.load(basepath+segfile))
            self.exit0.load_state_dict(torch.load(basepath+exitfile))
        elif ei == 1:
            self.seg1.load_state_dict(torch.load(basepath+segfile))
            self.exit1.load_state_dict(torch.load(basepath+exitfile))
        elif ei == 2:
            self.seg2.load_state_dict(torch.load(basepath+segfile))
            self.exit2.load_state_dict(torch.load(basepath+exitfile))
        else:
            print(f"Cannot load exit {ei}")
    
    def set_timing(enable=False):
        self.time_enable = enable

    def save_exits(self, basepath="modelexits/"):
        for ei in range(self.myexit+1):
            segfile= f"qseg{ei}.pt"
            exitfile= f"qexit{ei}.pt"
            #fetch correct statedict
            if ei == 0:
                torch.save(self.seg0.state_dict(), basepath + segfile)
                torch.save(self.exit0.state_dict(), basepath + exitfile)
            elif ei == 1:
                torch.save(self.seg1.state_dict(), basepath + segfile)
                torch.save(self.exit1.state_dict(), basepath + exitfile)
            elif ei == 2:
                torch.save(self.seg2.state_dict(), basepath + segfile)
                torch.save(self.exit2.state_dict(), basepath + exitfile)
            else:
                print(f"Cannot save exit {ei}")
    
    def setexit(self, myexit):
        self.myexit = myexit
        self.freeze_exits()

    def freeze_exits(self):
        print(f"Freeze Exits {self.myexit}")
        freeze = (self.myexit > 0)
        for p in self.seg0.parameters():
            p.requires_grad = not freeze
        for p in self.exit0.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.seg0.eval()
            self.exit0.eval()

        freeze = (self.myexit > 1)
        for p in self.seg1.parameters():
            p.requires_grad = not freeze
        for p in self.exit1.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.seg1.eval()
            self.exit1.eval()

        freeze = (self.myexit > 2)
        for p in self.seg2.parameters():
            p.requires_grad = not freeze
        for p in self.exit2.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.seg1.eval()
            self.exit1.eval()


def cnv(cfg, exits=-1):
    weight_bit_width = cfg.getint("QUANT", "WEIGHT_BIT_WIDTH")
    act_bit_width = cfg.getint("QUANT", "ACT_BIT_WIDTH")
    in_bit_width = cfg.getint("QUANT", "IN_BIT_WIDTH")
    num_classes = cfg.getint("MODEL", "NUM_CLASSES")
    in_channels = cfg.getint("MODEL", "IN_CHANNELS")
    if exits == -1:
        return CNV(
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            in_bit_width=in_bit_width,
            num_classes=num_classes,
            in_ch=in_channels,
        )
    elif exits == 0:
         return CNV_E0(
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            in_bit_width=in_bit_width,
            num_classes=num_classes,
            in_ch=in_channels,
        )

    return None
