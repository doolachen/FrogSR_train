import abc
import copy

import torch
import torch.nn as nn


class CompressAlgo(metaclass=abc.ABCMeta):
    def __init__(self):
        self.calibrate_enable = False
        self.compress_enable = False

    def enable_compress(self):
        self.calibrate_enable = True

    def disable_compress(self):
        self.calibrate_enable = False

    def enable_calibrate(self):
        self.compress_enable = True

    def disable_calibrate(self):
        self.compress_enable = False

    def compress(self, x):
        """
          Compressed data.
          Parameters:
          x: torch.Tensor, shape: [m, n_extracted_features]
          Return:
          x: torch.Tensor, shape: [m, n_compressed_features]
        """
        if self.calibrate_enable:
            self.calibrate_impl(x)
        if self.compress_enable:
            x = self.compress_impl(x)
        return x

    def extract(self, x):
        """
          Extract from compressed data.
          Parameters:
          x: torch.Tensor, shape: [m, n_compressed_features]
          Return:
          x: torch.Tensor, shape: [m, n_extracted_features]
        """
        if self.compress_enable:
            x = self.extract_impl(x)
        return x

    @abc.abstractmethod
    def compress_impl(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @abc.abstractmethod
    def extract_impl(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @abc.abstractmethod
    def calibrate_impl(self, x: torch.Tensor):
        pass

    @abc.abstractmethod
    def convert_to_compressor(self) -> nn.Module:
        return nn.Module()

    @abc.abstractmethod
    def convert_to_extractor(self) -> nn.Module:
        return nn.Module()


class CompressStub(nn.Module):
    def __init__(self, algo: CompressAlgo):
        super().__init__()
        self.algo = algo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.algo.compress(x)

    def convert(self):
        return self.algo.convert_to_compressor()


class ExtractStub(nn.Module):
    def __init__(self, algo: CompressAlgo):
        super().__init__()
        self.algo = algo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.algo.extract(x)

    def convert(self):
        return self.algo.convert_to_extractor()


def stub_pair(algo: CompressAlgo) -> (CompressStub, ExtractStub):
    return CompressStub(algo), ExtractStub(algo)


def enable_calibrate(module: nn.Module) -> nn.Module:
    for name, mod in module.named_children():
        if isinstance(mod, CompressStub) or isinstance(mod, ExtractStub):
            mod.algo.enable_calibrate()
        elif isinstance(mod, nn.Module):
            enable_calibrate(mod)
    return module


def disable_calibrate(module: nn.Module) -> nn.Module:
    for name, mod in module.named_children():
        if isinstance(mod, CompressStub) or isinstance(mod, ExtractStub):
            mod.algo.disable_calibrate()
        elif isinstance(mod, nn.Module):
            disable_calibrate(mod)
    return module


def enable_compress(module: nn.Module) -> nn.Module:
    for name, mod in module.named_children():
        if isinstance(mod, CompressStub) or isinstance(mod, ExtractStub):
            mod.algo.enable_compress()
        elif isinstance(mod, nn.Module):
            enable_compress(mod)
    return module


def disable_compress(module: nn.Module) -> nn.Module:
    for name, mod in module.named_children():
        if isinstance(mod, CompressStub) or isinstance(mod, ExtractStub):
            mod.algo.disable_compress()
        elif isinstance(mod, nn.Module):
            disable_compress(mod)
    return module


def convert(module: nn.Module, inplace=True) -> nn.Module:
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        if isinstance(mod, CompressStub) or isinstance(mod, ExtractStub):
            reassign[name] = mod.convert()
        elif isinstance(mod, nn.Module):
            convert(mod, inplace=True)
    for key, value in reassign.items():
        module._modules[key] = value
    return module
