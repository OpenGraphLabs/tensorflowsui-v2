// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

module tensorflowsui::layer {
    use std::string;
    use tensorflowsui::tensor;

    public struct Layer has copy, drop, store {
        layer_type: string::String,
        in_dimension: u64,
        out_dimension: u64,
        weight_tensor: tensor::SignedFixedTensor,
        bias_tensor: tensor::SignedFixedTensor,
    }

    public fun new_layer(
        layer_type: string::String,
        in_dimension: u64,
        out_dimension: u64,
        weights_magnitudes: vector<u64>,
        weights_signs: vector<u64>,
        biases_magnitudes: vector<u64>,
        biases_signs: vector<u64>,
        scale: u64,
    ): Layer {
        let weight_tensor = tensor::create_signed_fixed_tensor(
          vector[in_dimension, out_dimension],
            weights_magnitudes,
            weights_signs,
            scale,
          );

        let bias_tensor = tensor::create_signed_fixed_tensor(
          vector[out_dimension], 
          biases_magnitudes, 
          biases_signs, 
          scale,
          );
          
        Layer {
            layer_type,
            in_dimension,
            out_dimension,
            weight_tensor,
            bias_tensor,
        }
    }
}