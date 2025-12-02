To see all deposits or withdrawals from an address:

```
query MyQuery {
    Deposit(
      where: {
        _or: [
          { sender: { _eq: "0x16388463d60ffe0661cf7f1f31a7d658ac790ff7" } }
          { owner: { _eq: "0x16388463d60ffe0661cf7f1f31a7d658ac790ff7" } }
        ]
      }
    ) {
      id
      sender
      owner
      assets
      shares
    }

    Withdraw(
      where: {
        _or: [
          { sender: { _eq: "0x16388463d60ffe0661cf7f1f31a7d658ac790ff7" } }
          { receiver: { _eq: "0x16388463d60ffe0661cf7f1f31a7d658ac790ff7" } }
          { owner: { _eq: "0x16388463d60ffe0661cf7f1f31a7d658ac790ff7" } }
        ]
      }
    ) {
      id
      sender
      receiver
      owner
      assets
      shares
    }
  }
```

To see all unique depositor addresses
```
query GetUniqueDepositors {
    Deposit(distinct_on: owner, order_by: { owner: asc }) {
      owner
    }
  }
```


