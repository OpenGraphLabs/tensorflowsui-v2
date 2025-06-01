// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

module tensorflowsui::allowlist {
  use sui::table;

  // Error codes
  const EAlreadyInAllowlist: u64 = 0;
  const ENoAccess: u64 = 1;
  const ENotInAllowlist: u64 = 2;

  public struct Allowlist has key {
    id: UID,
    dataset_id: ID,
    addresses: table::Table<address, bool>,
  }

  public struct ALLOWLIST has drop {}

  public struct AllowlistManagerCap has key {
      id: UID
  }

  public fun new_allowlist_manager_cap(_witness: ALLOWLIST, ctx: &mut TxContext): AllowlistManagerCap {
      AllowlistManagerCap { id: object::new(ctx) }        
  }

  fun init(witness: ALLOWLIST, ctx: &mut TxContext) {
      let cap = new_allowlist_manager_cap(witness, ctx);
      transfer::transfer(cap, tx_context::sender(ctx));
  }

  public(package) fun new_allowlist(dataset_id: ID, ctx: &mut TxContext) {
      let al = create_allowlist(dataset_id, ctx);
      transfer::share_object(al);
  }
  
  public(package) fun create_allowlist(dataset_id: ID, ctx: &mut TxContext): Allowlist {
      let mut al = Allowlist {
          id: object::new(ctx),
          dataset_id,
          addresses: table::new(ctx),
      };
      al.addresses.add(ctx.sender(), true);

      al
  }

  public(package) fun add(al: &mut Allowlist, _: &AllowlistManagerCap, account: address) {
      assert!(!al.addresses.contains(account), EAlreadyInAllowlist);
      al.addresses.add(account, true);
  }

  public(package) fun remove(al: &mut Allowlist, _: &AllowlistManagerCap, account: address) {
      assert!(al.addresses.contains(account), ENotInAllowlist);
      al.addresses.remove(account);
  }

  //////////////////////////////////////////////////////////
  /// Access control
  /// key format: [pkg id][creator address][random nonce]

  /// All allowlisted addresses can access all IDs with the prefix of the whitelist
  fun check_policy(caller: address, id: vector<u8>, al: &Allowlist): bool {
      // Check if the id has the right prefix
      let prefix = al.id.to_bytes();
      let mut i = 0;
      if (prefix.length() > id.length()) {
          return false
      };
      while (i < prefix.length()) {
          if (prefix[i] != id[i]) {
              return false
          };
          i = i + 1;
      };

      // Check if user is in the allowlist
      al.addresses.contains(caller)
  }

  entry fun seal_approve(
    id: vector<u8>,
    al: &Allowlist,
    ctx: &TxContext,
  ) {
    assert!(check_policy(ctx.sender(), id, al), ENoAccess);
  }
}