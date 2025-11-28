## Yearn V3 Vault Indexer

This repository contains an Envio indexer for Yearn V3 vaults and a fee calculator script to analyze depositor positions.

*Please refer to the [documentation website](https://docs.envio.dev) for a thorough guide on all [Envio](https://envio.dev) indexer features*

### Debugging

1. Start docker desktop
2. Run `ps auxf | grep docker-proxy` and then `sudo kill 1234` to manually kill the processes with docker-proxy
3. Stop and remove the current docker containers
```
docker stop generated-graphql-engine-1 generated-envio-postgres-1
docker rm generated-graphql-engine-1 generated-envio-postgres-1
```
4. Now run envio with `pnpm dev`

### Pre-requisites

- [Node.js (use v18 or newer)](https://nodejs.org/en/download/current)
- [pnpm (use v8 or newer)](https://pnpm.io/installation)
- [Docker desktop](https://www.docker.com/products/docker-desktop/)

### Setup

```bash
# Install dependencies
pnpm install
```

### Run the Indexer

```bash
pnpm dev
```

Visit http://localhost:8080 to see the GraphQL Playground, local password is `testing`.

### Calculate Depositor Fees

Once the indexer is running, use the Python calculator to analyze any depositor:

```bash
python3 scripts/calc_depositor_fees.py <depositor-address>
```

**Example:**
```bash
python3 scripts/calc_depositor_fees.py 0x93A62dA5a14C80f265DAbC077fCEE437B1a0Efde
```

The script now validates that the performance fee stays constant across the depositor's entire history and that the management fee remains zero. It samples five even-spaced blocks between the first and last event, checks the fee configuration via the accountant contract, and stops with a clear error if anything changed so you can trust the rest of the calculation.

**Output includes:**
- Complete list of deposits, withdrawals, and transfers
- Current position (shares and value)
- Total profit/loss to date
- Estimated fees paid
- Complete event timeline
- Debugging information for fee calculation verification

**Environment Variables:**
- `ENVIO_GRAPHQL_URL` - GraphQL endpoint (default: `http://localhost:8080/v1/graphql`)
- `ENVIO_PASSWORD` - GraphQL password (default: `testing`)
- `RPC_URL` - Ethereum RPC endpoint for current state queries (default: `https://eth.merkle.io`)

### Generate files from `config.yaml` or `schema.graphql`

```bash
pnpm codegen
```
