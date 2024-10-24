#pragma once
#include "../common.hpp"

namespace pvfinder {

void leakyReLU(Tensor& x, float alpha = 0.01f);
void softplus(Tensor& x);
void scale(Tensor& x, float factor);
void concatenate(Tensor& dest, const Tensor& src);
Tensor reshapeToUNet(const Tensor& input);

} // namespace pvfinder
