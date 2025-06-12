#[test_only]
module tensorflowsui::tensor_tests {
    use sui::test_utils::{assert_eq};
    use std::debug;
    use std::string;
    use tensorflowsui::tensor;
    use tensorflowsui::math;

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

    #[test]
    fun test_various_scales() {
        // Test scale 2 (hundredths)
        let shape = vector[2];
        let mag_a = vector[125, 250];  // 1.25, 2.50
        let sign_a = vector[0, 0];
        let mag_b = vector[75, 150];   // 0.75, 1.50
        let sign_b = vector[0, 0];
        
        let tensor_a_scale2 = tensor::new_tensor(shape, mag_a, sign_a, 2);
        let tensor_b_scale2 = tensor::new_tensor(shape, mag_b, sign_b, 2);
        let result_scale2 = tensor::add(&tensor_a_scale2, &tensor_b_scale2);
        
        let result_mag = tensor::get_magnitude(&result_scale2);
        assert!(*vector::borrow(&result_mag, 0) == 200, 5001); // 1.25 + 0.75 = 2.00
        assert!(*vector::borrow(&result_mag, 1) == 400, 5002); // 2.50 + 1.50 = 4.00
        
        // Test scale 3 (thousandths)
        let mag_a3 = vector[1250, 2500];  // 1.250, 2.500
        let mag_b3 = vector[750, 1500];   // 0.750, 1.500
        
        let tensor_a_scale3 = tensor::new_tensor(shape, mag_a3, sign_a, 3);
        let tensor_b_scale3 = tensor::new_tensor(shape, mag_b3, sign_b, 3);
        let result_scale3 = tensor::add(&tensor_a_scale3, &tensor_b_scale3);
        
        let result_mag3 = tensor::get_magnitude(&result_scale3);
        assert!(*vector::borrow(&result_mag3, 0) == 2000, 5003); // 1.250 + 0.750 = 2.000
        assert!(*vector::borrow(&result_mag3, 1) == 4000, 5004); // 2.500 + 1.500 = 4.000
        
        // Test scale 4 (ten-thousandths)
        let mag_a4 = vector[12500, 25000];  // 1.2500, 2.5000
        let mag_b4 = vector[7500, 15000];   // 0.7500, 1.5000
        
        let tensor_a_scale4 = tensor::new_tensor(shape, mag_a4, sign_a, 4);
        let tensor_b_scale4 = tensor::new_tensor(shape, mag_b4, sign_b, 4);
        let result_scale4 = tensor::add(&tensor_a_scale4, &tensor_b_scale4);
        
        let result_mag4 = tensor::get_magnitude(&result_scale4);
        assert!(*vector::borrow(&result_mag4, 0) == 20000, 5005); // 1.2500 + 0.7500 = 2.0000
        assert!(*vector::borrow(&result_mag4, 1) == 40000, 5006); // 2.5000 + 1.5000 = 4.0000
    }

    #[test]
    fun test_scale_limits() {
        // Test maximum practical scale values
        let shape = vector[1];
        let mag = vector[1];
        let sign = vector[0];
        
        // Test scale 6 (common for financial calculations)
        let tensor_scale6 = tensor::new_tensor(shape, vector[1000000], sign, 6); // 1.000000
        assert!(tensor::get_scale(&tensor_scale6) == 6, 5007);
        
        // Test scale 8
        let tensor_scale8 = tensor::new_tensor(shape, vector[100000000], sign, 8); // 1.00000000
        assert!(tensor::get_scale(&tensor_scale8) == 8, 5008);
        
        // Test scale 10
        let tensor_scale10 = tensor::new_tensor(shape, vector[10000000000], sign, 10); // 1.0000000000
        assert!(tensor::get_scale(&tensor_scale10) == 10, 5009);
    }

    #[test]
    fun test_cross_scale_multiplication() {
        // Test multiplication with same scale maintains precision
        let shape = vector[2];
        
        // Scale 3: 1.500 * 2.000 = 3.000
        let mag_a = vector[1500, 3000];  // 1.500, 3.000
        let sign_a = vector[0, 0];
        let mag_b = vector[2000, 1000];  // 2.000, 1.000
        let sign_b = vector[0, 0];
        
        let tensor_a = tensor::new_tensor(shape, mag_a, sign_a, 3);
        let tensor_b = tensor::new_tensor(shape, mag_b, sign_b, 3);
        let result = tensor::multiply(&tensor_a, &tensor_b);
        
        let result_mag = tensor::get_magnitude(&result);
        // 1.500 * 2.000 = 3.000 (with scale 3 = 3000)
        assert!(*vector::borrow(&result_mag, 0) == 3000, 5010);
        // 3.000 * 1.000 = 3.000 (with scale 3 = 3000)
        assert!(*vector::borrow(&result_mag, 1) == 3000, 5011);
    }

    #[test]
    fun test_maximum_scale_boundary() {
        let shape = vector[1];
        let mag = vector[1];
        let sign = vector[0];
        
        // Test scale 15 - should work
        let tensor_scale15 = tensor::new_tensor(shape, mag, sign, 15);
        assert!(tensor::get_scale(&tensor_scale15) == 15, 5020);
        
        // Test scale 18 - should work (10^18 = 1,000,000,000,000,000,000)
        let tensor_scale18 = tensor::new_tensor(shape, mag, sign, 18);
        assert!(tensor::get_scale(&tensor_scale18) == 18, 5021);
    }
    
    #[test] 
    #[expected_failure(abort_code = 3002)]
    fun test_scale_overflow_boundary() {
        let shape = vector[1];
        let mag = vector[1];
        let sign = vector[0];
        
        // Test scale 25 - should fail when to_string calls math::scale_up
        // 10^25 would definitely exceed u64 max (18,446,744,073,709,551,615)
        let tensor_scale25 = tensor::new_tensor(shape, mag, sign, 25);
        // This should trigger the overflow in math::scale_up
        let _output = tensor::to_string(&tensor_scale25);
    }

    #[test]
    fun test_scale_computational_efficiency() {
        // This test demonstrates the computational difference between scales
        // Higher scales require more loop iterations in scale_up function
        
        let shape = vector[3];
        let mag = vector[100, 200, 300];
        let sign = vector[0, 0, 0];
        
        // Scale 2: requires 2 multiplications per scale_up call
        let tensor_scale2 = tensor::new_tensor(shape, mag, sign, 2);
        let _output2 = tensor::to_string(&tensor_scale2); // Triggers scale_up(1, 2)
        
        // Scale 5: requires 5 multiplications per scale_up call  
        let tensor_scale5 = tensor::new_tensor(shape, mag, sign, 5);
        let _output5 = tensor::to_string(&tensor_scale5); // Triggers scale_up(1, 5)
        
        // Scale 10: requires 10 multiplications per scale_up call
        let tensor_scale10 = tensor::new_tensor(shape, mag, sign, 10);
        let _output10 = tensor::to_string(&tensor_scale10); // Triggers scale_up(1, 10)
        
        // Scale 15: requires 15 multiplications per scale_up call
        let tensor_scale15 = tensor::new_tensor(shape, mag, sign, 15);
        let _output15 = tensor::to_string(&tensor_scale15); // Triggers scale_up(1, 15)
        
        // Note: Each scale value N requires N loop iterations in scale_up()
        // So scale 15 uses ~7.5x more computational operations than scale 2
    }

    #[test]
    fun test_multiplication_scale_cost() {
        // Test how scale affects multiplication operations specifically
        let shape = vector[2];
        let mag_a = vector[1000, 2000];
        let sign_a = vector[0, 0];
        let mag_b = vector[1500, 500];
        let sign_b = vector[0, 0];
        
        // Scale 2: multiply calls scale_up(1, 2) -> 2 iterations
        let a_scale2 = tensor::new_tensor(shape, mag_a, sign_a, 2);
        let b_scale2 = tensor::new_tensor(shape, mag_b, sign_b, 2);
        let _result2 = tensor::multiply(&a_scale2, &b_scale2);
        
        // Scale 8: multiply calls scale_up(1, 8) -> 8 iterations  
        let a_scale8 = tensor::new_tensor(shape, mag_a, sign_a, 8);
        let b_scale8 = tensor::new_tensor(shape, mag_b, sign_b, 8);
        let _result8 = tensor::multiply(&a_scale8, &b_scale8);
        
        // Scale 8 uses 4x more computational operations than scale 2
        // Gas consumption: scale8 ≈ 4x scale2 for the scale_up calls
    }

    #[test]
    fun test_division_scale_cost() {
        // Division is more expensive - it calls scale_up twice!
        let shape = vector[2];
        let mag_a = vector[2000, 4000]; 
        let sign_a = vector[0, 0];
        let mag_b = vector[1000, 2000];
        let sign_b = vector[0, 0];
        
        // Scale 3: division calls scale_up(1, 3) -> 3 iterations
        let a_scale3 = tensor::new_tensor(shape, mag_a, sign_a, 3);
        let b_scale3 = tensor::new_tensor(shape, mag_b, sign_b, 3);
        let _result3 = tensor::divide(&a_scale3, &b_scale3);
        
        // Scale 12: division calls scale_up(1, 12) -> 12 iterations
        let a_scale12 = tensor::new_tensor(shape, mag_a, sign_a, 12);
        let b_scale12 = tensor::new_tensor(shape, mag_b, sign_b, 12);
        let _result12 = tensor::divide(&a_scale12, &b_scale12);
        
        // Scale 12 uses 4x more computational operations than scale 3
        // Division is most expensive operation due to scaling overhead
    }

    #[test]
    fun test_scale_efficiency_comparison() {
        // Compare original scale_up vs optimized version
        
        // Original: O(n) complexity - gas consumption increases linearly with scale
        let original_scale2 = math::scale_up(1, 2);    // 2 loop iterations
        let original_scale10 = math::scale_up(1, 10);  // 10 loop iterations  
        let original_scale18 = math::scale_up(1, 18);  // 18 loop iterations
        
        // Optimized: O(1) complexity - constant gas consumption regardless of scale
        let optimized_scale2 = math::scale_up_optimized(1, 2);   // 1 lookup operation
        let optimized_scale10 = math::scale_up_optimized(1, 10); // 1 lookup operation
        let optimized_scale18 = math::scale_up_optimized(1, 18); // 1 lookup operation
        
        // Verify results are identical
        assert!(original_scale2 == optimized_scale2, 6001);   // Both = 100
        assert!(original_scale10 == optimized_scale10, 6002); // Both = 10,000,000,000
        assert!(original_scale18 == optimized_scale18, 6003); // Both = 1,000,000,000,000,000,000
        
        // Gas consumption patterns:
        // Original scale_up: scale2 < scale10 < scale18 (proportional to scale)
        // Optimized scale_up: scale2 ≈ scale10 ≈ scale18 (constant)
    }

    #[test]
    fun test_linear_to_constant_gas_conversion() {
        // Demonstrate gas conversion from O(n) to O(1)
        let shape = vector[2];
        let mag_a = vector[1000, 2000];
        let sign_a = vector[0, 0];
        let mag_b = vector[500, 1500];
        let sign_b = vector[0, 0];
        
        // ===== ORIGINAL FUNCTIONS: O(scale) Linear Gas =====
        
        // Scale 2: Original multiply - 2 iterations in scale_up
        let a2_orig = tensor::new_tensor(shape, mag_a, sign_a, 2);
        let b2_orig = tensor::new_tensor(shape, mag_b, sign_b, 2);
        let _result2_orig = tensor::multiply(&a2_orig, &b2_orig);
        // Gas: ~2x base cost
        
        // Scale 10: Original multiply - 10 iterations in scale_up  
        let a10_orig = tensor::new_tensor(shape, mag_a, sign_a, 10);
        let b10_orig = tensor::new_tensor(shape, mag_b, sign_b, 10);
        let _result10_orig = tensor::multiply(&a10_orig, &b10_orig);
        // Gas: ~10x base cost (5x more than scale 2!)
        
        // Scale 18: Original multiply - 18 iterations in scale_up
        let a18_orig = tensor::new_tensor(shape, mag_a, sign_a, 18);
        let b18_orig = tensor::new_tensor(shape, mag_b, sign_b, 18);
        let _result18_orig = tensor::multiply(&a18_orig, &b18_orig);
        // Gas: ~18x base cost (9x more than scale 2!)
        
        // ===== OPTIMIZED FUNCTIONS: O(1) Constant Gas =====
        
        // Scale 2: Optimized multiply - 1 lookup in scale_up_optimized
        let a2_opt = tensor::new_tensor(shape, mag_a, sign_a, 2);
        let b2_opt = tensor::new_tensor(shape, mag_b, sign_b, 2);
        let result2_opt = tensor::multiply_optimized(&a2_opt, &b2_opt);
        // Gas: ~1x base cost
        
        // Scale 10: Optimized multiply - 1 lookup in scale_up_optimized
        let a10_opt = tensor::new_tensor(shape, mag_a, sign_a, 10);
        let b10_opt = tensor::new_tensor(shape, mag_b, sign_b, 10);
        let result10_opt = tensor::multiply_optimized(&a10_opt, &b10_opt);
        // Gas: ~1x base cost (SAME as scale 2!)
        
        // Scale 18: Optimized multiply - 1 lookup in scale_up_optimized
        let a18_opt = tensor::new_tensor(shape, mag_a, sign_a, 18);
        let b18_opt = tensor::new_tensor(shape, mag_b, sign_b, 18);
        let result18_opt = tensor::multiply_optimized(&a18_opt, &b18_opt);
        // Gas: ~1x base cost (SAME as scale 2!)
        
        // Verify results are mathematically identical
        assert!(tensor::get_magnitude(&result2_opt) == tensor::get_magnitude(&_result2_orig), 7001);
        assert!(tensor::get_magnitude(&result10_opt) == tensor::get_magnitude(&_result10_orig), 7002);
        assert!(tensor::get_magnitude(&result18_opt) == tensor::get_magnitude(&_result18_orig), 7003);
        
        // Gas Consumption Summary:
        // BEFORE: scale2(2x) < scale10(10x) < scale18(18x) - LINEAR SCALING
        // AFTER:  scale2(1x) ≈ scale10(1x) ≈ scale18(1x) - CONSTANT SCALING
    }

    #[test]
    fun test_string_conversion_gas_improvement() {
        // String conversion shows the most dramatic improvement
        let shape = vector[3];
        let mag = vector[12345, 67890, 11111];
        let sign = vector[0, 1, 0];
        
        // Original: O(scale) per tensor element
        let tensor_scale15_orig = tensor::new_tensor(shape, mag, sign, 15);
        let _output_orig = tensor::to_string(&tensor_scale15_orig);
        // Gas: 15 iterations × 3 elements = 45x base cost
        
        // Optimized: O(1) per tensor element  
        let tensor_scale15_opt = tensor::new_tensor(shape, mag, sign, 15);
        let output_opt = tensor::to_string_optimized(&tensor_scale15_opt);
        // Gas: 1 lookup × 3 elements = 3x base cost (15x improvement!)
        
        // Results are identical
        assert!(_output_orig == output_opt, 7004);
        
        // For scale 15 with 3 elements:
        // BEFORE: 45x base gas cost (15 × 3)
        // AFTER:  3x base gas cost (1 × 3)  
        // IMPROVEMENT: 93% gas reduction!
    }

    #[test]
    fun test_division_gas_optimization() {
        // Division benefits significantly from optimization
        let shape = vector[2];
        let mag_a = vector[5000, 8000];
        let sign_a = vector[0, 0];
        let mag_b = vector[1000, 2000];
        let sign_b = vector[0, 0];
        
        // Test extreme scale difference
        let scale_low = 3;   // Original: 3 iterations
        let scale_high = 15; // Original: 15 iterations
        
        // Low scale comparison
        let a_low_orig = tensor::new_tensor(shape, mag_a, sign_a, scale_low);
        let b_low_orig = tensor::new_tensor(shape, mag_b, sign_b, scale_low);
        let _result_low_orig = tensor::divide(&a_low_orig, &b_low_orig);
        
        let a_low_opt = tensor::new_tensor(shape, mag_a, sign_a, scale_low);
        let b_low_opt = tensor::new_tensor(shape, mag_b, sign_b, scale_low);
        let result_low_opt = tensor::divide_optimized(&a_low_opt, &b_low_opt);
        
        // High scale comparison
        let a_high_orig = tensor::new_tensor(shape, mag_a, sign_a, scale_high);
        let b_high_orig = tensor::new_tensor(shape, mag_b, sign_b, scale_high);
        let _result_high_orig = tensor::divide(&a_high_orig, &b_high_orig);
        
        let a_high_opt = tensor::new_tensor(shape, mag_a, sign_a, scale_high);
        let b_high_opt = tensor::new_tensor(shape, mag_b, sign_b, scale_high);
        let result_high_opt = tensor::divide_optimized(&a_high_opt, &b_high_opt);
        
        // Verify correctness
        assert!(tensor::get_magnitude(&result_low_opt) == tensor::get_magnitude(&_result_low_orig), 7005);
        assert!(tensor::get_magnitude(&result_high_opt) == tensor::get_magnitude(&_result_high_orig), 7006);
        
        // Gas analysis:
        // Original scale 3:  3x gas
        // Original scale 15: 15x gas (5x more expensive)
        // Optimized scale 3:  1x gas  
        // Optimized scale 15: 1x gas (SAME cost!)
        // High scale improvement: 93% gas reduction (15x → 1x)
    }
} 