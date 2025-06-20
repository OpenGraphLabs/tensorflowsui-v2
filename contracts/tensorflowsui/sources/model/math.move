// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

module tensorflowsui::math {
  /// @notice Helper function to add two signed values
  /// @param sign1 Sign of first value (0: positive, 1: negative)
  /// @param magnitude1 Magnitude of first value
  /// @param sign2 Sign of second value (0: positive, 1: negative)
  /// @param magnitude2 Magnitude of second value
  /// @return Tuple of (result_sign, result_magnitude)
  public fun add_signed_number(
      s1: u64, m1: u64,
      s2: u64, m2: u64
  ): (u64, u64) {
    if (s1 == s2) {
      // Same sign: add magnitudes
      // Overflow protection: check if m1 + m2 would overflow
      assert!(m1 <= (18446744073709551615u64 - m2), 3001); // Prevent overflow
      (s1, m1 + m2)
    } else {
      // Different signs: subtract magnitudes
      if (m1 >= m2) {
        // First value has larger magnitude, keep its sign
        (s1, m1 - m2)
      } else {
        // Second value has larger magnitude, use its sign
        (s2, m2 - m1)
      }
    }
  }


  /// @notice Compares two signed numbers and determines if first is greater than second
  /// @param a_sign Sign of first number (0: positive, 1: negative)
  /// @param a_mag Magnitude of first number
  /// @param b_sign Sign of second number (0: positive, 1: negative)
  /// @param b_mag Magnitude of second number
  /// @return true if first number is greater than second number
  public fun compare_signed_number(a_sign: u64, a_mag: u64, b_sign: u64, b_mag: u64): bool {
    // If signs are different, positive is always greater than negative
    if (a_sign != b_sign) {
        return a_sign == 0
    };
    
    // For same signs
    if (a_sign == 0) {
        // Both positive: compare magnitudes directly
        a_mag > b_mag
    } else {
        // Both negative: compare magnitudes inversely
        a_mag < b_mag
    }
  }

  /// @notice Helper function to convert scale to a factor
  /// @param scale Scale value
  /// @return 10^scale value
  public fun get_scale_factor(scale: u64): u64 {
    let mut factor = 1;
    let mut i = 0;
    while (i < scale) {
        // Overflow protection: check if factor * 10 would overflow
        assert!(factor <= (18446744073709551615u64 / 10), 3003); // Prevent overflow
        factor = factor * 10;
        i = i + 1;
    };
    factor
  }

  /// @notice Scales up a value by a given scale
  /// @param value The value to scale up
  /// @param scale The scale to apply
  /// @return The scaled up value
  public fun scale_up(value: u64, scale:u64): u64 {
    let mut result = value;
    let mut i = 0;

    while (i < scale) {
        // Overflow protection: check if result * 10 would overflow
        assert!(result <= (18446744073709551615u64 / 10), 3002); // Prevent overflow
        result = result * 10;
        i = i + 1;
    };
    result
  }

  /// @notice Optimized scale function using lookup table (much faster!)
  /// @param value The value to scale up
  /// @param scale The scale to apply (0-18 only)
  /// @return The scaled up value
  public fun scale_up_optimized(value: u64, scale: u64): u64 {
    // Pre-computed powers of 10 for O(1) lookup instead of O(n) loop
    let scale_factors = vector[
        1,                     // 10^0
        10,                    // 10^1
        100,                   // 10^2
        1000,                  // 10^3
        10000,                 // 10^4
        100000,                // 10^5
        1000000,               // 10^6
        10000000,              // 10^7
        100000000,             // 10^8
        1000000000,            // 10^9
        10000000000,           // 10^10
        100000000000,          // 10^11
        1000000000000,         // 10^12
        10000000000000,        // 10^13
        100000000000000,       // 10^14
        1000000000000000,      // 10^15
        10000000000000000,     // 10^16
        100000000000000000,    // 10^17
        1000000000000000000    // 10^18 (max safe for u64)
    ];
    
    assert!(scale < vector::length(&scale_factors), 3004); // Scale too large
    
    let factor = *vector::borrow(&scale_factors, scale);
    
    // Overflow protection
    assert!(value == 0 || factor <= (18446744073709551615u64 / value), 3005);
    
    value * factor
  }
}
