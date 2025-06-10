#[test_only]
module tensorflowsui::tensor_tests {
    use sui::test_utils::{assert_eq};
    use std::debug;
    use std::string;
    use tensorflowsui::tensor;

    // Helper function to create a simple 1D tensor for testing
    fun create_test_tensor(values: vector<u64>, signs: vector<u64>, scale: u64): tensor::Tensor {
        let shape = vector[vector::length(&values)];
        tensor::new_tensor(shape, values, signs, scale)
    }

    // Helper function to create a 2D tensor for testing
    fun create_test_tensor_2d(rows: u64, cols: u64, values: vector<u64>, signs: vector<u64>, scale: u64): tensor::Tensor {
        let shape = vector[rows, cols];
        tensor::new_tensor(shape, values, signs, scale)
    }

    // Helper function to validate tensor results
    fun validate_tensor_result(
        result: &tensor::Tensor,
        expected_magnitude: vector<u64>,
        expected_sign: vector<u64>,
        expected_scale: u64
    ) {
        assert_eq(tensor::get_magnitude(result), expected_magnitude);
        assert_eq(tensor::get_sign(result), expected_sign);
        assert_eq(tensor::get_scale(result), expected_scale);
    }

    #[test]
    fun test_tensor_addition_basic() {
        // Test: [1.5, 2.0] + [0.5, 1.0] = [2.0, 3.0]
        // Scale = 1: 15 + 5 = 20, 20 + 10 = 30
        let tensor_a = create_test_tensor(
            vector[15, 20], // 1.5, 2.0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[5, 10],  // 0.5, 1.0 with scale=1
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::add(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[20, 30], vector[0, 0], 1);
    }

    #[test]
    fun test_tensor_addition_with_negatives() {
        // Test: [1.5, -2.0] + [-0.5, 1.0] = [1.0, -1.0]
        // Scale = 1: (0,15) + (1,5) = (0,10), (1,20) + (0,10) = (1,10)
        let tensor_a = create_test_tensor(
            vector[15, 20], // 1.5, -2.0 with scale=1
            vector[0, 1],   // positive, negative
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[5, 10],  // -0.5, 1.0 with scale=1
            vector[1, 0],   // negative, positive
            1
        );

        let result = tensor::add(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[10, 10], vector[0, 1], 1);
    }

    #[test]
    fun test_tensor_addition_zeros() {
        // Test: [0, 5.0] + [0, 0] = [0, 5.0]
        let tensor_a = create_test_tensor(
            vector[0, 50],  // 0, 5.0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[0, 0],   // 0, 0
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::add(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[0, 50], vector[0, 0], 1);
    }

    #[test]
    fun test_tensor_addition_2d() {
        // Test 2x2 matrix addition: [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
        let tensor_a = create_test_tensor_2d(
            2, 2,
            vector[10, 20, 30, 40], // scale=1
            vector[0, 0, 0, 0],
            1
        );
        
        let tensor_b = create_test_tensor_2d(
            2, 2,
            vector[50, 60, 70, 80], // scale=1
            vector[0, 0, 0, 0],
            1
        );

        let result = tensor::add(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[60, 80, 100, 120], vector[0, 0, 0, 0], 1);
    }

    #[test]
    fun test_tensor_subtraction_basic() {
        // Test: [3.0, 5.0] - [1.0, 2.0] = [2.0, 3.0]
        // Scale = 1: 30 - 10 = 20, 50 - 20 = 30
        let tensor_a = create_test_tensor(
            vector[30, 50], // 3.0, 5.0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[10, 20], // 1.0, 2.0 with scale=1
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::subtract(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[20, 30], vector[0, 0], 1);
    }

    #[test]
    fun test_tensor_subtraction_with_negatives() {
        // Test: [1.0, -2.0] - [-3.0, 1.0] = [4.0, -3.0]
        // (0,10) - (1,30) = (0,10) + (0,30) = (0,40)
        // (1,20) - (0,10) = (1,20) + (1,10) = (1,30)
        let tensor_a = create_test_tensor(
            vector[10, 20], // 1.0, -2.0 with scale=1
            vector[0, 1],   // positive, negative
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[30, 10], // -3.0, 1.0 with scale=1
            vector[1, 0],   // negative, positive
            1
        );

        let result = tensor::subtract(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[40, 30], vector[0, 1], 1);
    }

    #[test]
    fun test_tensor_subtraction_result_negative() {
        // Test: [1.0, 2.0] - [3.0, 5.0] = [-2.0, -3.0]
        let tensor_a = create_test_tensor(
            vector[10, 20], // 1.0, 2.0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[30, 50], // 3.0, 5.0 with scale=1
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::subtract(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[20, 30], vector[1, 1], 1);
    }

    #[test]
    fun test_tensor_multiplication_basic() {
        // Test: [2.0, 3.0] * [1.5, 2.0] = [3.0, 6.0]
        // Scale = 1: (20 * 15) / 10 = 30, (30 * 20) / 10 = 60
        let tensor_a = create_test_tensor(
            vector[20, 30], // 2.0, 3.0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[15, 20], // 1.5, 2.0 with scale=1
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::multiply(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[30, 60], vector[0, 0], 1);
    }

    #[test]
    fun test_tensor_multiplication_with_negatives() {
        // Test: [2.0, -3.0] * [-1.0, 2.0] = [-2.0, -6.0]
        // Sign rules: pos*neg=neg, neg*pos=neg
        let tensor_a = create_test_tensor(
            vector[20, 30], // 2.0, -3.0 with scale=1
            vector[0, 1],   // positive, negative
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[10, 20], // -1.0, 2.0 with scale=1
            vector[1, 0],   // negative, positive
            1
        );

        let result = tensor::multiply(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[20, 60], vector[1, 1], 1);
    }

    #[test]
    fun test_tensor_multiplication_with_zeros() {
        // Test: [2.0, 0] * [3.0, 5.0] = [6.0, 0]
        let tensor_a = create_test_tensor(
            vector[20, 0],  // 2.0, 0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[30, 50], // 3.0, 5.0 with scale=1
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::multiply(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[60, 0], vector[0, 0], 1);
    }

    #[test]
    fun test_tensor_division_basic() {
        // Test: [6.0, 9.0] / [2.0, 3.0] = [3.0, 3.0]
        // Scale = 1: (60 * 10) / 20 = 30, (90 * 10) / 30 = 30
        let tensor_a = create_test_tensor(
            vector[60, 90], // 6.0, 9.0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[20, 30], // 2.0, 3.0 with scale=1
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::divide(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[30, 30], vector[0, 0], 1);
    }

    #[test]
    fun test_tensor_division_with_negatives() {
        // Test: [-6.0, 8.0] / [2.0, -4.0] = [-3.0, -2.0]
        let tensor_a = create_test_tensor(
            vector[60, 80], // -6.0, 8.0 with scale=1
            vector[1, 0],   // negative, positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[20, 40], // 2.0, -4.0 with scale=1
            vector[0, 1],   // positive, negative
            1
        );

        let result = tensor::divide(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[30, 20], vector[1, 1], 1);
    }

    #[test]
    fun test_tensor_division_fractional_result() {
        // Test: [5.0, 3.0] / [2.0, 4.0] = [2.5, 0.75]
        // Scale = 1: (50 * 10) / 20 = 25, (30 * 10) / 40 = 7 (integer division)
        let tensor_a = create_test_tensor(
            vector[50, 30], // 5.0, 3.0 with scale=1
            vector[0, 0],   // both positive
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[20, 40], // 2.0, 4.0 with scale=1
            vector[0, 0],   // both positive
            1
        );

        let result = tensor::divide(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[25, 7], vector[0, 0], 1);
    }

    #[test]
    fun test_higher_precision_arithmetic() {
        // Test with scale=2 (2 decimal places): [1.25, 3.75] + [2.50, 1.25] = [3.75, 5.00]
        let tensor_a = create_test_tensor(
            vector[125, 375], // 1.25, 3.75 with scale=2
            vector[0, 0],     // both positive
            2
        );
        
        let tensor_b = create_test_tensor(
            vector[250, 125], // 2.50, 1.25 with scale=2
            vector[0, 0],     // both positive
            2
        );

        let result = tensor::add(&tensor_a, &tensor_b);
        validate_tensor_result(&result, vector[375, 500], vector[0, 0], 2);

        // Test multiplication: scale should stay 2
        let mult_result = tensor::multiply(&tensor_a, &tensor_b);
        validate_tensor_result(&mult_result, vector[312, 468], vector[0, 0], 2);
    }

    #[test]
    fun test_edge_case_large_numbers() {
        // Test with larger numbers to ensure no overflow in reasonable ranges
        let tensor_a = create_test_tensor(
            vector[1000000, 2000000], // 10000.00, 20000.00 with scale=2
            vector[0, 0],             // both positive
            2
        );
        
        let tensor_b = create_test_tensor(
            vector[500000, 300000],   // 5000.00, 3000.00 with scale=2
            vector[0, 0],             // both positive
            2
        );

        let add_result = tensor::add(&tensor_a, &tensor_b);
        validate_tensor_result(&add_result, vector[1500000, 2300000], vector[0, 0], 2);

        let sub_result = tensor::subtract(&tensor_a, &tensor_b);
        validate_tensor_result(&sub_result, vector[500000, 1700000], vector[0, 0], 2);
    }

    #[test]
    fun test_tensor_operations_consistency() {
        // Test that (a + b) - b = a
        let tensor_a = create_test_tensor(
            vector[123, 456], 
            vector[0, 1],     // mixed signs
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[789, 234], 
            vector[1, 0],     // mixed signs
            1
        );

        let sum = tensor::add(&tensor_a, &tensor_b);
        let difference = tensor::subtract(&sum, &tensor_b);
        
        // Should equal original tensor_a
        validate_tensor_result(&difference, vector[123, 456], vector[0, 1], 1);
    }

    #[test] 
    fun test_multiplicative_identity() {
        // Test that a * 1 = a (scale preserved)
        let tensor_a = create_test_tensor(
            vector[123, 456], 
            vector[0, 1],     // mixed signs
            1
        );
        
        let tensor_one = create_test_tensor(
            vector[10, 10],   // 1.0, 1.0 with scale=1
            vector[0, 0],     // both positive
            1
        );

        let result = tensor::multiply(&tensor_a, &tensor_one);
        // Result should be same magnitude and scale
        validate_tensor_result(&result, vector[123, 456], vector[0, 1], 1);
    }

    #[test]
    fun test_division_inverse() {
        // Test that (a / b) * b = a (with preserved scales)
        let tensor_a = create_test_tensor(
            vector[40, 60], // 4.0, 6.0 with scale=1
            vector[0, 0],     
            1
        );
        
        let tensor_b = create_test_tensor(
            vector[20, 20],   // 2.0, 2.0 with scale=1
            vector[0, 0],     
            1
        );

        let division_result = tensor::divide(&tensor_a, &tensor_b);
        validate_tensor_result(&division_result, vector[20, 30], vector[0, 0], 1);
        
        let multiply_back = tensor::multiply(&division_result, &tensor_b);
        // Now both operations preserve scale, so result should match original
        validate_tensor_result(&multiply_back, vector[40, 60], vector[0, 0], 1);
    }

    // Error handling tests - using actual numeric error codes from tensor.move
    #[test]
    #[expected_failure(abort_code = 1001)]
    fun test_add_different_scales_should_fail() {
        let tensor_a = create_test_tensor(vector[100], vector[0], 1);
        let tensor_b = create_test_tensor(vector[100], vector[0], 2);
        
        tensor::add(&tensor_a, &tensor_b); // Should abort with 1001
    }

    #[test]
    #[expected_failure(abort_code = 1101)]
    fun test_subtract_different_scales_should_fail() {
        let tensor_a = create_test_tensor(vector[100], vector[0], 1);
        let tensor_b = create_test_tensor(vector[100], vector[0], 2);
        
        tensor::subtract(&tensor_a, &tensor_b); // Should abort with 1101
    }

    #[test]
    #[expected_failure(abort_code = 1201)]
    fun test_multiply_different_scales_should_fail() {
        let tensor_a = create_test_tensor(vector[100], vector[0], 1);
        let tensor_b = create_test_tensor(vector[100], vector[0], 2);
        
        tensor::multiply(&tensor_a, &tensor_b); // Should abort with 1201
    }

    #[test]
    #[expected_failure(abort_code = 1301)]
    fun test_divide_different_scales_should_fail() {
        let tensor_a = create_test_tensor(vector[100], vector[0], 1);
        let tensor_b = create_test_tensor(vector[100], vector[0], 2);
        
        tensor::divide(&tensor_a, &tensor_b); // Should abort with 1301
    }

    #[test]
    #[expected_failure(abort_code = 1002)]
    fun test_add_different_lengths_should_fail() {
        let tensor_a = create_test_tensor(vector[100, 200], vector[0, 0], 1);
        let tensor_b = create_test_tensor(vector[100], vector[0], 1);
        
        tensor::add(&tensor_a, &tensor_b); // Should abort with 1002
    }

    #[test]
    #[expected_failure(abort_code = 9999)]
    fun test_divide_by_zero_should_fail() {
        let tensor_a = create_test_tensor(vector[100], vector[0], 1);
        let tensor_b = create_test_tensor(vector[0], vector[0], 1);
        
        tensor::divide(&tensor_a, &tensor_b); // Should abort with 9999
    }

    #[test]
    fun test_debug_tensor_operations() {
        // This test demonstrates tensor operations with debug output
        debug::print(&string::utf8(b"=== Tensor Operations Debug Test ==="));
        
        let tensor_a = create_test_tensor(
            vector[125, 250], // 1.25, 2.50 with scale=2
            vector[0, 1],     // positive, negative
            2
        );
        
        let tensor_b = create_test_tensor(
            vector[75, 150],  // 0.75, 1.50 with scale=2
            vector[1, 0],     // negative, positive
            2
        );

        debug::print(&string::utf8(b"Tensor A:"));
        tensor::debug_print_tensor(&tensor_a);
        
        debug::print(&string::utf8(b"Tensor B:"));
        tensor::debug_print_tensor(&tensor_b);

        let add_result = tensor::add(&tensor_a, &tensor_b);
        debug::print(&string::utf8(b"A + B:"));
        tensor::debug_print_tensor(&add_result);

        let mult_result = tensor::multiply(&tensor_a, &tensor_b);
        debug::print(&string::utf8(b"A * B:"));
        tensor::debug_print_tensor(&mult_result);
    }

    #[test]
    fun test_safe_chained_operations() {
        let tensor_a = create_test_tensor(vector[250, 150], vector[0, 0], 2); // 2.50, 1.50
        
        // Chain: a * a + a = a² + a
        let squared = tensor::multiply(&tensor_a, &tensor_a);   // Should stay scale=2
        let result = tensor::add(&squared, &tensor_a);         // Should work!
        
        // Verify scales match for chaining
        assert_eq(tensor::get_scale(&squared), 2);
        assert_eq(tensor::get_scale(&result), 2);
        
        // Verify mathematical correctness: 2.5² + 2.5 = 6.25 + 2.5 = 8.75 → 875
        // squared: (250*250)/100 = 625, result: 625 + 250 = 875
        validate_tensor_result(&result, vector[875, 375], vector[0, 0], 2);
    }
} 