# Yearn V3 Depositor Fee Tracker - Price Per Share Differential Method
## Implementation Specification

---

## Overview

This specification describes how to calculate the exact value of fees accumulated by a specific depositor address in a Yearn V3 vault (e.g., yvUSDC) using the **Price Per Share Differential Method**.

### Core Principle

Fees in Yearn V3 are charged by minting new vault shares, which dilutes existing shareholders. This method calculates fees by comparing:
1. **Theoretical Value**: What the depositor's shares would be worth if no fees were charged
2. **Actual Value**: What the depositor's shares are actually worth with fees charged

The difference represents the total fees paid by the depositor.

---

## Architecture: Using Envio for Off-Chain Data Collection

This implementation leverages the **Envio indexer** infrastructure that already exists in this repository. The architecture consists of two main components:

### 1. Envio Indexer (Off-Chain Data Collection)
- **Purpose**: Automatically collects and indexes vault events from the blockchain
- **Components**:
  - `config.yaml`: Defines which contracts and events to index
  - `src/EventHandlers.ts`: Processes events and stores them in the database
  - `schema.graphql`: Defines the data structure for indexed events
  - PostgreSQL database: Stores indexed event data
  - GraphQL API: Provides query interface at http://localhost:8080

### 2. Fee Calculation Engine (This Specification)
- **Purpose**: Queries indexed events and calculates fees paid by a depositor
- **Data Sources**:
  - **Envio GraphQL API**: For historical event data (Deposits, Withdrawals, Transfers, StrategyReported)
  - **Archive Node RPC**: For historical state queries (only when needed for pricePerShare, totalSupply, etc.)

**Benefits of This Architecture:**
- Event data is collected once and stored locally (no repeated RPC calls)
- Fast GraphQL queries instead of slow event log filtering
- Real-time indexing keeps data up-to-date
- Reduced dependency on archive nodes (only needed for state queries)

---

## Prerequisites

### Required Knowledge
- GraphQL query syntax
- ERC-4626 vault standard
- Basic understanding of Yearn V3 architecture
- Event log parsing
- Understanding of Envio indexer architecture

### Required Infrastructure
- **Envio Indexer**: This repository includes an Envio indexer that automatically collects and indexes vault events
- **Node.js 18+** and **pnpm**
- **Docker Desktop**: Required for running the Envio indexer locally
- **Archive Node Access** (optional): Only needed for real-time state queries (Alchemy, Infura, QuickNode, or self-hosted)

### Required Contract Addresses
You will need:
1. Yearn V3 Vault address (e.g., yvUSDC vault)
2. Vault's Accountant contract address (obtained from vault)
3. VaultFactory address (for protocol fee config)
4. Underlying asset address (e.g., USDC)

---

## Envio Indexer Setup

### Step 0: Running the Envio Indexer

Before calculating fees, you need to ensure the Envio indexer is running and has indexed the relevant vault events.

**Starting the Indexer:**
```bash
# Install dependencies
pnpm install

# Start the indexer (includes GraphQL API on http://localhost:8080)
pnpm dev
```

**Verify Indexer Status:**
Visit http://localhost:8080 to access the GraphQL Playground (password: `testing`)

**Check Indexed Events:**
```graphql
query {
  Deposit(limit: 5, orderBy: { id: "desc" }) {
    items {
      id
      sender
      owner
      assets
      shares
    }
  }
}
```

### Indexed Event Types

The Envio indexer in this repository automatically collects the following events from the Yearn V3 vault:

- **Deposit**: User deposits into the vault
- **Withdraw**: User withdrawals from the vault
- **Transfer**: Share transfers between addresses
- **StrategyReported**: Strategy profit/loss reports with fee information
- **UpdateAccountant**: Changes to the accountant contract
- And various other vault configuration events

All events are stored in a PostgreSQL database and queryable via GraphQL API at `http://localhost:8080`.

---

## Data Collection Phase

### Step 1: Collect Depositor Transaction History

**Objective**: Build a complete timeline of the depositor's interactions with the vault using the Envio GraphQL API.

#### GraphQL Queries for Events

**1.1 Deposit Events**

Query the Envio indexer for all deposits made by a specific address:

```graphql
query GetDepositorDeposits($depositorAddress: String!) {
  Deposit(
    where: { owner: { eq: $depositorAddress } }
    orderBy: { id: "asc" }
  ) {
    items {
      id
      sender
      owner
      assets
      shares
    }
  }
}
```

**Data Available**:
- `id`: Unique identifier in format `{chainId}_{blockNumber}_{logIndex}`
- `owner`: Address that owns the deposited shares
- `sender`: Address that initiated the deposit
- `assets`: Amount of underlying asset deposited
- `shares`: Vault shares received
- `price_per_share_at_deposit`: Can be calculated as `assets / shares`

**1.2 Withdrawal Events**

Query for all withdrawals by a specific address:

```graphql
query GetDepositorWithdrawals($depositorAddress: String!) {
  Withdraw(
    where: { owner: { eq: $depositorAddress } }
    orderBy: { id: "asc" }
  ) {
    items {
      id
      sender
      receiver
      owner
      assets
      shares
    }
  }
}
```

**Data Available**:
- `id`: Unique identifier with block and log index information
- `owner`: Address that owns the shares being withdrawn
- `receiver`: Address receiving the underlying assets
- `assets`: Amount of underlying asset withdrawn
- `shares`: Vault shares burned
- `price_per_share_at_withdrawal`: Can be calculated as `assets / shares`

**1.3 Transfer Events (for share movements)**

Query for share transfers involving the depositor:

```graphql
query GetDepositorTransfers($depositorAddress: String!) {
  transfersFrom: Transfer(
    where: { sender: { eq: $depositorAddress } }
    orderBy: { id: "asc" }
  ) {
    items {
      id
      sender
      receiver
      value
    }
  }

  transfersTo: Transfer(
    where: { receiver: { eq: $depositorAddress } }
    orderBy: { id: "asc" }
  ) {
    items {
      id
      sender
      receiver
      value
    }
  }
}
```

**Purpose**: Detect if depositor transferred shares to another address (reduces their exposure) or received shares from another address.

**Note**: The `id` field format (`{chainId}_{blockNumber}_{logIndex}`) contains the block number, which can be parsed to determine the block when each event occurred.

---

### Step 2: Collect Vault Strategy Report Events

**Objective**: Identify all times when strategies reported profits/losses and fees were charged using the Envio GraphQL API.

#### GraphQL Query for Strategy Reports

Query all strategy reports from the vault:

```graphql
query GetStrategyReports {
  StrategyReported(
    orderBy: { id: "asc" }
  ) {
    items {
      id
      strategy
      gain
      loss
      current_debt
      protocol_fees
      total_fees
      total_refunds
    }
  }
}
```

**Data Available**:
- `id`: Unique identifier in format `{chainId}_{blockNumber}_{logIndex}` (contains block number)
- `strategy`: Strategy address that reported
- `gain`: Profit generated since last report (in underlying asset units)
- `loss`: Loss incurred since last report (in underlying asset units)
- `current_debt`: Current amount of vault assets deployed to this strategy
- `protocol_fees`: Fees paid to Yearn protocol (in vault shares)
- `total_fees`: Total fees charged (in vault shares)
- `total_refunds`: Refunds given back (in underlying asset units)

**Important Notes**:
- `total_fees` includes both protocol fees AND performance fees paid to vault manager
- Fees (`protocol_fees` and `total_fees`) are denominated in vault shares, not underlying asset
- `gain` is the gross profit before fees
- The block number can be extracted from the `id` field for timeline ordering

---

### Step 3: Query Fee Configuration

**Objective**: Understand the fee structure to reconstruct theoretical no-fee scenarios.

#### 3.1 Accountant Contract Fee Configuration

**Contract Call**: `accountant.feeConfig(vault_address)`

Typical return structure:
```solidity
struct FeeConfig {
    uint16 managementFee;      // Annual management fee in basis points
    uint16 performanceFee;     // Performance fee in basis points
    uint16 refundRatio;        // Ratio for fee refunds
    uint16 maxFee;             // Maximum fee cap
    uint16 maxGain;            // Maximum gain to charge fees on
    uint16 maxLoss;            // Maximum loss to recognize
}
```

**Data to Extract**:
- `management_fee_bps`: Annual management fee (e.g., 200 = 2%)
- `performance_fee_bps`: Performance fee on gains (e.g., 1000 = 10%)

**Note**: Management fees are typically charged over time, performance fees on strategy gains.

#### 3.2 Protocol Fee Configuration

**Contract**: VaultFactory address
**Function**: `protocol_fee_config(vault_address)` or `protocol_fee_config()` for default

**Returns**: `protocol_fee_bps` (basis points, e.g., 1000 = 10% of total fees)

**Important**: Protocol fees are a percentage OF the total fees, not a separate fee on gains.

**Example Calculation**:
```
Gain = 100 USDC
Performance Fee = 10% = 10 USDC (in fee shares)
Protocol Fee = 10% of 10 = 1 USDC (of the fee shares)
Vault Manager Fee = 9 USDC (of the fee shares)
```

---

### Step 4: Query Historical State at Key Blocks

**Objective**: Get point-in-time state of the vault and depositor's position.

**Note**: While the Envio indexer provides event data, you still need RPC access (archive node) for historical state queries at specific blocks.

#### Required State Queries

For each deposit, withdrawal, and strategy report block, query using viem or ethers.js:

**4.1 Vault State**
```typescript
import { createPublicClient, http } from 'viem';
import { mainnet } from 'viem/chains';

const client = createPublicClient({
  chain: mainnet,
  transport: http('YOUR_ARCHIVE_NODE_URL'),
});

// At specific block_number:
const [pricePerShare, totalSupply, totalAssets, totalDebt, totalIdle] =
  await Promise.all([
    client.readContract({
      address: vaultAddress,
      abi: vaultAbi,
      functionName: 'pricePerShare',
      blockNumber: blockNumber,
    }),
    client.readContract({
      address: vaultAddress,
      abi: vaultAbi,
      functionName: 'totalSupply',
      blockNumber: blockNumber,
    }),
    client.readContract({
      address: vaultAddress,
      abi: vaultAbi,
      functionName: 'totalAssets',
      blockNumber: blockNumber,
    }),
    // ... additional calls
  ]);
```

**4.2 Depositor State**
```typescript
// At specific block_number:
const depositorBalance = await client.readContract({
  address: vaultAddress,
  abi: vaultAbi,
  functionName: 'balanceOf',
  args: [depositorAddress],
  blockNumber: blockNumber,
});

const assetsValue = await client.readContract({
  address: vaultAddress,
  abi: vaultAbi,
  functionName: 'convertToAssets',
  args: [depositorBalance],
  blockNumber: blockNumber,
});
```

**4.3 Strategy State (for each strategy)**
```typescript
// Query strategy state at specific block
const strategyAssets = await client.readContract({
  address: strategyAddress,
  abi: strategyAbi,
  functionName: 'totalAssets',
  blockNumber: blockNumber,
});
```

---

## Calculation Phase

### Step 5: Build Depositor Position Timeline

**Objective**: Create a chronological record of the depositor's position changes.

#### Data Structure

```typescript
interface PositionSnapshot {
  blockNumber: number;
  timestamp: Date;
  eventType: string; // 'deposit', 'withdraw', 'transfer', 'report'

  // Depositor state
  sharesBalance: bigint;
  sharesChange: bigint; // Delta from previous snapshot
  assetsDeposited: bigint; // Cumulative
  assetsWithdrawn: bigint; // Cumulative

  // Vault state
  totalSupply: bigint;
  totalAssets: bigint;
  pricePerShare: bigint;

  // Ownership
  ownershipPercentage: bigint; // shares_balance / total_supply (in basis points)

  // For report events
  gain: bigint;
  loss: bigint;
  totalFeesShares: bigint;
  protocolFeesShares: bigint;
}

interface DepositorPosition {
  address: string;
  snapshots: PositionSnapshot[];
  currentShares: bigint;
  totalDeposited: bigint;
  totalWithdrawn: bigint;
}
```

**Note**: This specification uses TypeScript with `bigint` for precision. The calculation algorithms shown in later sections use Python-style pseudocode for clarity, but should be implemented with proper big number handling in your chosen language (TypeScript/JavaScript should use `bigint`, Python should use `Decimal`, etc.).

#### Algorithm

```typescript
interface DepositEvent {
  id: string;
  owner: string;
  sender: string;
  assets: bigint;
  shares: bigint;
}

interface WithdrawEvent {
  id: string;
  owner: string;
  receiver: string;
  sender: string;
  assets: bigint;
  shares: bigint;
}

interface TransferEvent {
  id: string;
  sender: string;
  receiver: string;
  value: bigint;
}

interface StrategyReportedEvent {
  id: string;
  strategy: string;
  gain: bigint;
  loss: bigint;
  protocol_fees: bigint;
  total_fees: bigint;
  total_refunds: bigint;
}

// Helper function to parse block number from Envio event ID
function parseBlockNumber(eventId: string): number {
  // ID format: {chainId}_{blockNumber}_{logIndex}
  const parts = eventId.split('_');
  return parseInt(parts[1]);
}

function parseLogIndex(eventId: string): number {
  const parts = eventId.split('_');
  return parseInt(parts[2]);
}

async function buildPositionTimeline(
  depositorAddress: string,
  deposits: DepositEvent[],
  withdrawals: WithdrawEvent[],
  transfers: TransferEvent[],
  reports: StrategyReportedEvent[],
  vaultClient: any, // viem PublicClient
  startBlock: number,
  endBlock: number
): Promise<DepositorPosition> {
  /**
   * Build a complete timeline of depositor's position changes.
   */

  // Combine all events and sort by block number and log index
  const allEvents: Array<{type: string, event: any}> = [
    ...deposits.map(e => ({ type: 'deposit', event: e })),
    ...withdrawals.map(e => ({ type: 'withdraw', event: e })),
    ...transfers.map(e => ({ type: 'transfer', event: e })),
    ...reports.map(e => ({ type: 'report', event: e })),
  ];

  allEvents.sort((a, b) => {
    const blockA = parseBlockNumber(a.event.id);
    const blockB = parseBlockNumber(b.event.id);
    if (blockA !== blockB) return blockA - blockB;
    return parseLogIndex(a.event.id) - parseLogIndex(b.event.id);
  });

  const snapshots: PositionSnapshot[] = [];
  let currentShares = 0n;
  let totalDeposited = 0n;
  let totalWithdrawn = 0n;

  for (const { type: eventType, event } of allEvents) {
    // Parse block number from event ID
    const block = BigInt(parseBlockNumber(event.id));

    // Query vault state at this block using viem
    const [totalSupply, totalAssets, pricePerShare, depositorShares] =
      await Promise.all([
        vaultClient.readContract({
          address: vaultAddress,
          abi: vaultAbi,
          functionName: 'totalSupply',
          blockNumber: block,
        }),
        vaultClient.readContract({
          address: vaultAddress,
          abi: vaultAbi,
          functionName: 'totalAssets',
          blockNumber: block,
        }),
        vaultClient.readContract({
          address: vaultAddress,
          abi: vaultAbi,
          functionName: 'pricePerShare',
          blockNumber: block,
        }),
        vaultClient.readContract({
          address: vaultAddress,
          abi: vaultAbi,
          functionName: 'balanceOf',
          args: [depositorAddress],
          blockNumber: block,
        }),
      ]);

    let sharesChange = 0n;

    if (eventType === 'deposit') {
      sharesChange = event.shares;
      currentShares += sharesChange;
      totalDeposited += event.assets;
    } else if (eventType === 'withdraw') {
      sharesChange = -event.shares;
      currentShares += sharesChange; // shares_change is negative
      totalWithdrawn += event.assets;
    } else if (eventType === 'transfer') {
      if (event.sender.toLowerCase() === depositorAddress.toLowerCase()) {
        sharesChange = -event.value;
      } else {
        // to_address is depositor
        sharesChange = event.value;
      }
      currentShares += sharesChange;
    }

    // Create snapshot
    const snapshot: PositionSnapshot = {
      blockNumber: Number(block),
      timestamp: await getBlockTimestamp(vaultClient, block),
      eventType,
      sharesBalance: depositorShares, // Use actual on-chain value
      sharesChange,
      assetsDeposited: totalDeposited,
      assetsWithdrawn: totalWithdrawn,
      totalSupply,
      totalAssets,
      pricePerShare,
      ownershipPercentage:
        totalSupply > 0n ? (depositorShares * 10000n) / totalSupply : 0n,
      // Add report-specific data
      gain: eventType === 'report' ? event.gain : 0n,
      loss: eventType === 'report' ? event.loss : 0n,
      totalFeesShares: eventType === 'report' ? event.total_fees : 0n,
      protocolFeesShares: eventType === 'report' ? event.protocol_fees : 0n,
    };

    snapshots.push(snapshot);
  }

  return {
    address: depositorAddress,
    snapshots,
    currentShares,
    totalDeposited,
    totalWithdrawn,
  };
}
```

### Python Fee Calculator Implementation (calc_depositor_fees.py)

The repository ships a concrete Python implementation (`scripts/calc_depositor_fees.py`) that mirrors the conceptual steps above. The following describes its behavior so it can be re-implemented in another environment by reading only this specification.

1. **Event Retrieval**
   * `get_deposit_events`, `get_withdraw_events`, and `get_transfer_events` query the Envio GraphQL API for the depositor's events (filtering `owner`, `sender`, `receiver`, respectively) and order them by `id` ascending. The queries return the raw fields (`id`, `assets`, `shares`, `value`) needed for subsequent calculations.
2. **Timeline Reconstruction**
   * `parse_event_id` extracts `(block_number, log_index)` from each Envio `id`.
   * `build_event_timeline` tags each result with a normalized `Event` type (`deposit`, `withdraw`, `transfer_in`, `transfer_out`), merges them into a single list, and sorts by `(block_number, log_index)` to guarantee chronological order.
3. **Position Snapshots**
   * `calculate_position` iterates the sorted events, keeping running totals:
     - `current_shares` tracks the depositor's share balance after each event.
     - `total_deposited` and `total_withdrawn` accumulate underlying asset volumes.
     - Each event appends a `PositionSnapshot` dataclass capturing `block_number`, `event_type`, `shares_balance`, `shares_change`, and total assets moved so far.
4. **Entry PPS Tracking**
   * `calculate_weighted_average_entry_pps` replays the timeline to derive the depositor's cost basis:
     - Deposits add their asset/share pairs.
     - Withdrawals and transfer-outs remove shares proportionally and deduct the matched asset basis to keep the average cost accurate.
     - Transfer-ins are valued at the block's `pricePerShare`, fetched via `get_price_per_share_at_block` (which caches RPC responses), before they augment the totals.
     - The final entry PPS = `total_assets / total_shares`, scaled to the vault's decimals.
5. **Profit & Fee Math**
   * `calculate_incremental_profit_and_fees` builds a profit series:
     - Iterates the `PositionSnapshot` list, fetching the `pricePerShare` at each block and computing `delta_pps`.
     - Profit increments by `previous_shares * delta_pps` so gains accrue between events.
     - After processing all snapshots, the script applies `(currentPPS - last_snapshot_pps)` to capture the latest change.
     - `get_performance_fee_rate` reads the accountant contract to obtain the performance fee (basis points). The script applies this rate to net profit to report gross profit and estimated fees.
6. **Output & Visualization**
   * `format_output` summarizes current shares, deposit/withdraw totals, PPS comparison, net/gross profit, and fees.
   * User events are listed chronologically using the already sorted timeline, describing each deposit, withdrawal, or transfer with block numbers and asset/share amounts.
   * `plot_balance_profit` samples up to 300 snapshots, plots a step chart for share balances, overlays incremental profit on a twin y-axis, caches block timestamps for reuse, and adds a secondary x-axis labeled with the year at the start of each period.
7. **Supporting Features**
   * Block-level RPC calls are cached for `pricePerShare` and timestamp lookups (`price_per_share_cache`, `block_timestamp_cache`) to limit archive-node load.
   * Environment variables `ENVIO_GRAPHQL_URL`, `ENVIO_PASSWORD`, and `RPC_URL` are used with sensible defaults (see README) when running the script directly (`python3 scripts/calc_depositor_fees.py <address>`).

These details ensure the Python script's logic—timeline assembly, cost-basis tracking, incremental profit, and output composition—can be reconstructed from the specification alone.

---

## Remaining Considerations

The sections listed below were part of an earlier theoretical breakdown but are now superseded by the Python implementation in `scripts/calc_depositor_fees.py`. Consult that script (the single source of truth referenced above) for the exact math, fee decomposition, and output semantics.
