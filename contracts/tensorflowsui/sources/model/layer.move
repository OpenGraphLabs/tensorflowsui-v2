// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

module tensorflowsui::layer {
    use std::string;
    use tensorflowsui::tensor;
    use tensorflowsui::math;

    // const NONE : u64= 0;
    const RELU : u64= 1;
    // const SOFTMAX : u64 = 2;

    const ERR_DIMENSION_MISMATCH: u64 = 10001;
    const ERR_BIAS_DIMENSION_MISMATCH: u64 = 10002;
    const ERR_SCALE_MISMATCH: u64 = 10003;
    const ERR_SCALE_MISMATCH_BIAS: u64 = 10004;

    public struct Layer has copy, drop, store {
        layer_type: string::String,
        in_dimension: u64,
        out_dimension: u64,
        weight_tensor: tensor::Tensor,
        bias_tensor: tensor::Tensor,
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
        let weight_tensor = tensor::new_tensor(
          vector[in_dimension, out_dimension],
            weights_magnitudes,
            weights_signs,
            scale,
          );

        let bias_tensor = tensor::new_tensor(
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

    /// @notice Performs dense layer computation with optional activation function
    /// @param input_tensor Input tensor (batch_size x input_dimension)
    /// @param weight_tensor Weight tensor (input_dimension x output_dimension)
    /// @param bias_tensor Bias tensor (output_dimension)
    /// @param activation_type Activation function to apply (0=None, 1=ReLU, 2=Softmax)
    /// @return Result tensor (batch_size x output_dimension)
    public fun compute_dense_layer(
        input_tensor: &tensor::Tensor,
        weight_tensor: &tensor::Tensor,
        bias_tensor: &tensor::Tensor,
        activation_type: u64
    ): tensor::Tensor {
        // 1. Extract tensor dimensions and validate
        let batch_size = *vector::borrow(&tensor::get_shape(input_tensor), 0);
        let input_dim = *vector::borrow(&tensor::get_shape(input_tensor), 1);
        let weight_input_dim = *vector::borrow(&tensor::get_shape(weight_tensor), 0);
        let output_dim = *vector::borrow(&tensor::get_shape(weight_tensor), 1);
        let bias_dim = *vector::borrow(&tensor::get_shape(bias_tensor), 0);

        // Validate dimensions match
        assert!(input_dim == weight_input_dim, ERR_DIMENSION_MISMATCH);
        assert!(output_dim == bias_dim, ERR_BIAS_DIMENSION_MISMATCH);

        // Validate scales match
        let scale = tensor::get_scale(input_tensor);
        assert!(scale == tensor::get_scale(weight_tensor), ERR_SCALE_MISMATCH);
        assert!(scale == tensor::get_scale(bias_tensor), ERR_SCALE_MISMATCH_BIAS);

        // 2. Prepare output tensor
        let output_shape = vector[batch_size, output_dim];

        let mut output_magnitude = vector::empty<u64>();
        let mut output_sign = vector::empty<u64>();

        // Pre-compute scale factor
        let scale_factor = math::get_scale_factor(scale);

        // 3. Compute output for each batch item and output dimension
        let mut batch_idx = 0;
        while (batch_idx < batch_size) {
            // Process each output neuron (dimension)
            let mut output_idx = 0;
            while (output_idx < output_dim) {
                // Initialize accumulators for weighted sum
                let mut acc_sign = 0; // 0: positive, 1: negative
                let mut acc_magnitude = 0;
                
                // Compute weighted sum across input dimensions
                let mut input_idx = 0;
                while (input_idx < input_dim) {
                    // Calculate flat indices for accessing 1D tensor storage
                    let input_flat_idx = batch_idx * input_dim + input_idx;
                    let weight_flat_idx = input_idx * output_dim + output_idx;
                    
                    // Extract input and weight values
                    let input_sign = *vector::borrow(&tensor::get_sign(input_tensor), input_flat_idx);
                    let input_magnitude = *vector::borrow(&tensor::get_magnitude(input_tensor), input_flat_idx);
                    let weight_sign = *vector::borrow(&tensor::get_sign(weight_tensor), weight_flat_idx);
                    let weight_magnitude = *vector::borrow(&tensor::get_magnitude(weight_tensor), weight_flat_idx);

                    // Perform signed multiplication (XOR for sign)
                    let product_sign = if (input_sign == weight_sign) { 0 } else { 1 };
                    let product_magnitude = input_magnitude * weight_magnitude;

                    // Scale down the product to match the correct scale
                    // Product is currently at scale^2, so divide by scale_factor to bring back to scale
                    let scaled_product_magnitude = product_magnitude / scale_factor;

                    // Add product to accumulator
                    let (new_acc_sign, new_acc_magnitude) = math::add_signed_number(
                        acc_sign, acc_magnitude,
                        product_sign, scaled_product_magnitude
                    );
                    acc_sign = new_acc_sign;
                    acc_magnitude = new_acc_magnitude;

                    input_idx = input_idx + 1;
                };

                // Add bias (no need to scale bias, it's already at the correct scale)
                let bias_sign = *vector::borrow(&tensor::get_sign(bias_tensor), output_idx);
                let bias_magnitude = *vector::borrow(&tensor::get_magnitude(bias_tensor), output_idx);
                
                // Add bias to accumulated value
                let (final_sign, final_magnitude) = math::add_signed_number(
                    acc_sign, acc_magnitude,
                    bias_sign, bias_magnitude
                );

                // Apply activation function if specified
                let mut result_sign = final_sign;
                let mut result_magnitude = final_magnitude;
                
                if (activation_type == RELU && result_sign == 1) {
                    // For ReLU, zero out negative values
                    result_sign = 0;
                    result_magnitude = 0;
                };
                // TODO: Softmax activation would be implemented here if needed

                // Result is already at the correct scale, no need to scale down again
                vector::push_back(&mut output_sign, result_sign);
                vector::push_back(&mut output_magnitude, result_magnitude);

                output_idx = output_idx + 1;
            };
            batch_idx = batch_idx + 1;
        };

        // Create and return result tensor
        tensor::new_tensor(output_shape, output_magnitude, output_sign, scale)
    }

    public fun ReLu(weighted_sum : u64): u64 {
        if (weighted_sum > 0) {
            weighted_sum
        } else {
            0
        }
    }

    public fun get_weight_tensor(layer: &Layer): &tensor::Tensor {
        &layer.weight_tensor
    }

    public fun get_bias_tensor(layer: &Layer): &tensor::Tensor {
        &layer.bias_tensor
    }

    public fun get_in_dimension(layer: &Layer): u64 {
        layer.in_dimension
    }

    public fun get_out_dimension(layer: &Layer): u64 {
        layer.out_dimension
    }
}