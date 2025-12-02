#!/usr/bin/env python3
"""Python port of the Yearn V3 depositor fee calculator."""

import base64
import datetime
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

HAS_MATPLOTLIB = plt is not None



# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_local_env(file_path: str) -> None:
    if not os.path.isfile(file_path):
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            trimmed = line.strip()
            if not trimmed or trimmed.startswith('#'):
                continue

            if '=' not in trimmed:
                continue

            key, value = trimmed.split('=', 1)
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            os.environ[key] = value


DOTENV_FILE = os.path.join(os.getcwd(), '.env.local')
load_local_env(DOTENV_FILE)

ENVIO_GRAPHQL_URL = os.environ.get('ENVIO_GRAPHQL_URL', 'http://localhost:8080/v1/graphql')
ENVIO_PASSWORD = os.environ.get('ENVIO_PASSWORD', 'testing')
VAULT_ADDRESS = '0xBe53A109B494E5c9f97b9Cd39Fe969BE68BF6204'
RPC_URL = os.environ.get('RPC_URL', 'https://eth.merkle.io')
PRICE_PER_SHARE_SELECTOR = '0x99530b06'
DECIMALS_SELECTOR = '0x313ce567'

price_per_share_cache: Dict[int, int] = {}
block_timestamp_cache: Dict[int, datetime.datetime] = {}


@dataclass
class DepositEvent:
    id: str
    sender: str
    owner: str
    assets: str
    shares: str


@dataclass
class WithdrawEvent:
    id: str
    sender: str
    receiver: str
    owner: str
    assets: str
    shares: str


@dataclass
class TransferEvent:
    id: str
    sender: str
    receiver: str
    value: str


@dataclass
class Event:
    type: str
    block_number: int
    log_index: int
    data: Dict[str, Any]


@dataclass
class PositionSnapshot:
    block_number: int
    event_type: str
    shares_balance: int
    shares_change: int
    assets_deposited: int
    assets_withdrawn: int


@dataclass
class PositionResult:
    snapshots: List[PositionSnapshot]
    current_shares: int
    total_deposited: int
    total_withdrawn: int
    user_events: List[Event]
    peak_shares: int
    peak_shares_block: int


def rpc_call(method: str, params: List[Any]) -> Any:
    payload = json.dumps({
        'jsonrpc': '2.0',
        'method': method,
        'params': params,
        'id': 1,
    }).encode('utf-8')
    headers = {'Content-Type': 'application/json'}

    request = urllib.request.Request(RPC_URL, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f'RPC call failed: {exc}')

    if 'error' in result:
        raise RuntimeError(f"RPC error: {result['error'].get('message')}")
    return result.get('result')


def contract_call(address: str, data: str, block_number: Optional[int] = None) -> str:
    params: List[Any] = [{'to': address, 'data': data}]
    params.append(f'0x{block_number:x}' if block_number is not None else 'latest')
    return rpc_call('eth_call', params)


def get_price_per_share_at_block(block_number: int) -> int:
    if block_number in price_per_share_cache:
        return price_per_share_cache[block_number]

    price_hex = contract_call(VAULT_ADDRESS, PRICE_PER_SHARE_SELECTOR, block_number)
    value = int(price_hex, 16)
    price_per_share_cache[block_number] = value
    return value


def get_block_timestamp(block_number: int) -> datetime.datetime:
    if block_number in block_timestamp_cache:
        return block_timestamp_cache[block_number]

    try:
        block = rpc_call('eth_getBlockByNumber', [f'0x{block_number:x}', False])
        timestamp = int(block['timestamp'], 16)
        result = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
    except Exception:
        try:
            current_block_hex = rpc_call('eth_blockNumber', [])
            current_block = int(current_block_hex, 16)
            current_time = datetime.datetime.now(datetime.timezone.utc)
            block_diff = current_block - block_number
            result = current_time - datetime.timedelta(seconds=block_diff * 12)
        except Exception:
            result = datetime.datetime.now(datetime.timezone.utc)

    block_timestamp_cache[block_number] = result
    return result


def format_date(value: datetime.datetime) -> str:
    return value.strftime('%Y-%m-%d %H:%M:%S UTC')


def format_units(value: int, decimals: int = 6) -> str:
    sign = '-' if value < 0 else ''
    absolute = abs(value)
    if decimals == 0:
        return f"{sign}{absolute}"
    divisor = 10 ** decimals
    whole = absolute // divisor
    fraction = absolute % divisor
    if fraction == 0:
        return f"{sign}{whole}"
    frac_str = str(fraction).rjust(decimals, '0').rstrip('0')
    return f"{sign}{whole}.{frac_str}"


def query_envio_graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps({'query': query, 'variables': variables}).encode('utf-8')
    auth = base64.b64encode(f":{ENVIO_PASSWORD}".encode('utf-8')).decode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {auth}',
    }
    request = urllib.request.Request(ENVIO_GRAPHQL_URL, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f'GraphQL query failed: {exc}')

    if 'errors' in result:
        raise RuntimeError(f"GraphQL errors: {result['errors']}")
    return result.get('data', {})


def get_deposit_events(depositor_address: str) -> List[DepositEvent]:
    query = textwrap.dedent('''
        query GetDepositorDeposits($depositorAddress: String!) {
          Deposit(
            where: { owner: { _eq: $depositorAddress } }
            order_by: { id: asc }
          ) {
            id
            sender
            owner
            assets
            shares
          }
        }
    ''')
    data = query_envio_graphql(query, {'depositorAddress': depositor_address.lower()})
    return [DepositEvent(**entry) for entry in data.get('Deposit', [])]


def get_withdraw_events(depositor_address: str) -> List[WithdrawEvent]:
    query = textwrap.dedent('''
        query GetDepositorWithdrawals($depositorAddress: String!) {
          Withdraw(
            where: { owner: { _eq: $depositorAddress } }
            order_by: { id: asc }
          ) {
            id
            sender
            receiver
            owner
            assets
            shares
          }
        }
    ''')
    data = query_envio_graphql(query, {'depositorAddress': depositor_address.lower()})
    return [WithdrawEvent(**entry) for entry in data.get('Withdraw', [])]


def get_transfer_events(depositor_address: str) -> List[TransferEvent]:
    zero_address = '0x' + '0' * 40
    query = textwrap.dedent('''
        query GetDepositorTransfers($depositorAddress: String!, $zeroAddress: String!) {
          transfersFrom: Transfer(
            where: {
              sender: { _eq: $depositorAddress }
              receiver: { _neq: $zeroAddress }
            }
            order_by: { id: asc }
          ) {
            id
            sender
            receiver
            value
          }
          transfersTo: Transfer(
            where: {
              receiver: { _eq: $depositorAddress }
              sender: { _neq: $zeroAddress }
            }
            order_by: { id: asc }
          ) {
            id
            sender
            receiver
            value
          }
        }
    ''')
    data = query_envio_graphql(query, {
        'depositorAddress': depositor_address.lower(),
        'zeroAddress': zero_address.lower(),
    })
    return [TransferEvent(**entry) for entry in data.get('transfersFrom', []) + data.get('transfersTo', [])]


def parse_event_id(event_id: str) -> Tuple[int, int]:
    parts = event_id.split('_')
    return int(parts[1]), int(parts[2])


def build_event_timeline(
    deposits: List[DepositEvent],
    withdrawals: List[WithdrawEvent],
    transfers: List[TransferEvent],
    depositor_address: str,
) -> List[Event]:
    events: List[Event] = []

    for deposit in deposits:
        block, log = parse_event_id(deposit.id)
        events.append(Event('deposit', block, log, deposit.__dict__))

    for withdrawal in withdrawals:
        block, log = parse_event_id(withdrawal.id)
        events.append(Event('withdraw', block, log, withdrawal.__dict__))

    for transfer in transfers:
        block, log = parse_event_id(transfer.id)
        event_type = 'transfer_out' if transfer.sender.lower() == depositor_address.lower() else 'transfer_in'
        events.append(Event(event_type, block, log, transfer.__dict__))

    events.sort(key=lambda itm: (itm.block_number, itm.log_index))
    return events


def calculate_position(events: List[Event], depositor_address: str) -> PositionResult:
    snapshots: List[PositionSnapshot] = []
    user_events: List[Event] = []
    current_shares = 0
    total_deposited = 0
    total_withdrawn = 0
    peak_shares = 0
    peak_shares_block = 0

    for event in events:
        shares_change = 0
        if event.type == 'deposit':
            shares_change = int(event.data['shares'])
            current_shares += shares_change
            total_deposited += int(event.data['assets'])
            user_events.append(event)
        elif event.type == 'withdraw':
            shares_change = -int(event.data['shares'])
            current_shares += shares_change
            total_withdrawn += int(event.data['assets'])
            user_events.append(event)
        elif event.type in ('transfer_in', 'transfer_out'):
            shares_change = int(event.data['value']) if event.type == 'transfer_in' else -int(event.data['value'])
            current_shares += shares_change
            user_events.append(event)

        if current_shares > peak_shares:
            peak_shares = current_shares
            peak_shares_block = event.block_number

        snapshots.append(PositionSnapshot(
            block_number=event.block_number,
            event_type=event.type,
            shares_balance=current_shares,
            shares_change=shares_change,
            assets_deposited=total_deposited,
            assets_withdrawn=total_withdrawn,
        ))

    return PositionResult(
        snapshots=snapshots,
        current_shares=current_shares,
        total_deposited=total_deposited,
        total_withdrawn=total_withdrawn,
        user_events=user_events,
        peak_shares=peak_shares,
        peak_shares_block=peak_shares_block,
    )


def calculate_weighted_average_entry_pps(events: List[Event], decimals: int) -> int:
    scale = 10 ** decimals
    total_assets = 0
    total_shares = 0

    for event in events:
        if event.type == 'deposit':
            shares = int(event.data['shares'])
            assets = int(event.data['assets'])
            total_shares += shares
            total_assets += assets
        elif event.type == 'withdraw':
            shares = int(event.data['shares'])
            if total_shares > 0:
                remove_shares = min(shares, total_shares)
                removed_assets = (total_assets * remove_shares) // total_shares
                total_shares -= remove_shares
                total_assets -= removed_assets
        elif event.type == 'transfer_in':
            shares = int(event.data['value'])
            pps = get_price_per_share_at_block(event.block_number)
            assets = shares * pps // scale
            total_shares += shares
            total_assets += assets
        elif event.type == 'transfer_out':
            shares = int(event.data['value'])
            if total_shares > 0:
                remove_shares = min(shares, total_shares)
                removed_assets = (total_assets * remove_shares) // total_shares
                total_shares -= remove_shares
                total_assets -= removed_assets

    if total_shares == 0:
        return 0

    return total_assets * 1000000 // total_shares


def read_accountant_fee_config(vault_address: str, block_number: Optional[int] = None) -> Tuple[int, int, int, int]:
    accountant_hex = contract_call(vault_address, '0x4fb3ccc5', block_number)
    accountant_address = '0x' + accountant_hex[-40:]
    selector = '0xde1eb9a3'
    vault_param = vault_address.lower().replace('0x', '').rjust(64, '0')
    config_hex = contract_call(accountant_address, selector + vault_param, block_number)
    if not config_hex:
        raise RuntimeError('Empty getVaultConfig response')
    hex_payload = (config_hex[2:] if config_hex.startswith('0x') else config_hex).ljust(64 * 4, '0')
    if len(hex_payload) < 64 * 4:
        raise RuntimeError(f'getVaultConfig returned {len(hex_payload)} hex characters')
    words = [int(hex_payload[i * 64:(i + 1) * 64], 16) for i in range(4)]
    return tuple(words)  # management_fee, performance_fee, _, max_fee


def verify_management_fee_zero(vault_address: str, blocks: List[int]) -> None:
    for block in blocks:
        management_fee, _, _, _ = read_accountant_fee_config(vault_address, block)
        if management_fee != 0:
            raise RuntimeError(
                f'Management fee non-zero ({management_fee}) detected at block {block}; expected 0'
            )


def get_performance_fee_rate(vault_address: str, block_number: Optional[int] = None, *, log: bool = True) -> int:
    try:
        management_fee, performance_fee, _, max_fee = read_accountant_fee_config(vault_address, block_number)
        if management_fee != 0:
            raise RuntimeError(f'Unexpected management fee {management_fee} (expected 0)')
        if max_fee == 0:
            raise RuntimeError('maxFee is zero')
        ratio_bps = performance_fee * 10000 // max_fee
        if log:
            print(f'Performance fee {ratio_bps / 100}% (calculated from on-chain)')
        return ratio_bps
    except Exception as exc:
        print(f'Warning: could not fetch performance fee rate ({exc}) – using default 10%')
        return 1000


def sample_fee_check_blocks(start_block: int, end_block: int, checks: int = 5) -> List[int]:
    if checks <= 1 or start_block == end_block:
        return [start_block] * checks

    span = end_block - start_block
    if span <= 0:
        return [start_block] * checks

    return [start_block + (span * i) // (checks - 1) for i in range(checks)]


def verify_performance_fee_stability(
    vault_address: str,
    first_block: Optional[int],
    last_block: Optional[int],
    reference_fee_bps: int,
    checks: int = 5,
    *,
    blocks_to_check: Optional[List[int]] = None,
) -> List[int]:
    if first_block is None or last_block is None:
        return []
    if checks <= 0:
        return []

    if blocks_to_check is None:
        blocks_to_check = sample_fee_check_blocks(first_block, last_block, checks)
    observed = []
    for block in blocks_to_check:
        fee = get_performance_fee_rate(vault_address, block, log=False)
        observed.append((block, fee))

    if any(fee != reference_fee_bps for _, fee in observed):
        details = ', '.join(
            f'Block {block}: {fee / 100:.2f}%' for block, fee in observed
        )
        raise RuntimeError(
            f'Performance fee changed during depositor activity; expected '
            f'{reference_fee_bps / 100:.2f}%, observed [{details}]'
        )
    return blocks_to_check


def calculate_incremental_profit_and_fees(
    snapshots: List[PositionSnapshot],
    performance_fee_bps: int,
    current_pps: int,
    current_shares: int,
    decimals: int,
) -> Dict[str, int]:
    scale = 10 ** decimals
    net_profit = 0
    previous_shares = 0
    previous_pps = get_price_per_share_at_block(snapshots[0].block_number) if snapshots else current_pps

    for snapshot in snapshots:
        snapshot_pps = get_price_per_share_at_block(snapshot.block_number)
        delta_pps = snapshot_pps - previous_pps
        net_profit += previous_shares * delta_pps // scale
        previous_shares = snapshot.shares_balance
        previous_pps = snapshot_pps

    net_profit += previous_shares * (current_pps - previous_pps) // scale

    basis_points = 10000
    gross_profit = net_profit
    total_fees = 0
    if net_profit > 0 and basis_points > performance_fee_bps:
        gross_profit = net_profit * basis_points // (basis_points - performance_fee_bps)
        total_fees = gross_profit - net_profit

    return {
        'net_profit': net_profit,
        'gross_profit': gross_profit,
        'total_fees': total_fees,
        'effective_shares': current_shares,
    }


def sample_series(series: List[Dict[str, int]], max_points: int) -> List[Dict[str, int]]:
    if len(series) <= max_points:
        return series
    step = len(series) / max_points
    sampled: List[Dict[str, int]] = []
    for i in range(max_points):
        index = min(int(i * step), len(series) - 1)
        sampled.append(series[index])
    return sampled


def prepare_balance_profit_series(
    snapshots: List[PositionSnapshot],
    decimals: int,
    current_pps: int,
    current_shares: int,
) -> List[Dict[str, int]]:
    if not snapshots:
        return []
    scale = 10 ** decimals
    series: List[Dict[str, int]] = []
    profit = 0
    previous_shares = 0
    previous_pps = get_price_per_share_at_block(snapshots[0].block_number)

    for snapshot in snapshots:
        snapshot_pps = get_price_per_share_at_block(snapshot.block_number)
        delta_pps = snapshot_pps - previous_pps
        profit += previous_shares * delta_pps // scale
        previous_shares = snapshot.shares_balance
        previous_pps = snapshot_pps
        series.append({
            'block': snapshot.block_number,
            'shares': snapshot.shares_balance,
            'profit': profit,
        })

    # Add profit from last snapshot to current state
    profit += previous_shares * (current_pps - previous_pps) // scale

    # Add current state as final data point
    # Fetch current block number
    try:
        current_block_hex = rpc_call('eth_blockNumber', [])
        current_block = int(current_block_hex, 16)
    except Exception:
        # If RPC call fails, use last snapshot block + offset as approximation
        current_block = snapshots[-1].block_number + 1000

    series.append({
        'block': current_block,
        'shares': current_shares,
        'profit': profit,
    })

    return series


def plot_balance_profit(
    snapshots: List[PositionSnapshot],
    decimals: int,
    current_pps: int,
    current_shares: int,
) -> None:
    if not snapshots:
        return
    if not HAS_MATPLOTLIB:  # pragma: no cover
        print('Install matplotlib (`pip install matplotlib`) to see the popup plot.')
        return
    series = prepare_balance_profit_series(snapshots, decimals, current_pps, current_shares)
    graph_series = sample_series(series, 300)
    blocks = [point['block'] for point in graph_series]
    shares = [point['shares'] / 1_000_000 for point in graph_series]
    profits = [point['profit'] / (10 ** decimals) for point in graph_series]

    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        fig.canvas.manager.set_window_title('Balance and profit over time')
    except AttributeError:
        try:
            fig.canvas.set_window_title('Balance and profit over time')
        except AttributeError:
            pass
    # Use a step chart so balances stay flat between on-chain events and only jump at the event block.
    shares_line, = ax.step(blocks, shares, label='Share balance', color='tab:blue', where='post')
    ax.set_xlabel('Block')
    ax.set_ylabel('Shares (USDC share units)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(alpha=0.3)

    dates = [get_block_timestamp(block) for block in blocks]

    # Create top x-axis with start and end dates
    if dates:
        start_date = dates[0].strftime('%d/%m/%Y')
        end_date = dates[-1].strftime('%d/%m/%Y')

        ax_dates = ax.twiny()
        ax_dates.set_xlim(ax.get_xlim())

        # Show start and end dates at the edges
        tick_positions = [blocks[0], blocks[-1]]
        tick_labels = [f'Start: {start_date}', f'End: {end_date}']

        # Add year markers if span crosses multiple years
        year_ticks: List[Tuple[int, str]] = []
        current_year: Optional[int] = None
        for block, date in zip(blocks, dates):
            year = date.year
            if current_year is None or year != current_year:
                # Don't add year tick if it's too close to start or end
                if block != blocks[0] and block != blocks[-1]:
                    year_ticks.append((block, str(year)))
                current_year = year

        if year_ticks:
            tick_positions.extend([block for block, _ in year_ticks])
            tick_labels.extend([year for _, year in year_ticks])

        ax_dates.set_xticks(tick_positions)
        ax_dates.set_xticklabels(tick_labels)
        ax_dates.set_xlabel('Date Range')
        ax_dates.xaxis.set_label_position('top')
        ax_dates.xaxis.set_ticks_position('top')
        ax_dates.spines['top'].set_position(('outward', 36))
        ax_dates.tick_params(axis='x', labelrotation=15, labelsize=9)

    ax2 = ax.twinx()
    profit_line, = ax2.plot(blocks, profits, label='Incremental profit (USDC)', color='tab:green')
    ax2.set_ylabel('Incremental profit (USDC)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    lines = [shares_line, profit_line]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper left')
    fig.suptitle('Yearn V3 depositor balance vs incremental profit')
    fig.tight_layout()
    plt.show()


def format_output(
    depositor_address: str,
    deposits: List[DepositEvent],
    withdrawals: List[WithdrawEvent],
    transfers: List[TransferEvent],
    position: PositionResult,
    current_value: int,
    weighted_avg_entry_pps: int,
    profit_and_fees: Dict[str, int],
    performance_fee_bps: int,
    current_pps: int,
    first_interaction_date: Optional[datetime.datetime],
    first_interaction_block: Optional[int],
    peak_value: Optional[int],
    peak_date: Optional[datetime.datetime],
    decimals: int,
) -> None:
    net_profit = profit_and_fees['net_profit']
    gross_profit = profit_and_fees['gross_profit']
    total_fees = profit_and_fees['total_fees']
    current_shares = position.current_shares
    total_deposited = position.total_deposited
    total_withdrawn = position.total_withdrawn
    net_deposited = total_deposited - total_withdrawn
    transfers_in = [t for t in transfers if t.receiver.lower() == depositor_address.lower()]
    transfers_out = [t for t in transfers if t.sender.lower() == depositor_address.lower()]
    net_transferred_shares = sum(int(t.value) for t in transfers_in) - sum(int(t.value) for t in transfers_out)

    print('\n' + '=' * 80)
    print('YEARN V3 DEPOSITOR FEE & PROFIT ANALYSIS')
    print('=' * 80)
    print('\n📊 DEPOSITOR INFORMATION')
    print('-' * 80)
    print(f'Address: {depositor_address}')
    print(f'Vault:   {VAULT_ADDRESS}')
    if first_interaction_block and first_interaction_date:
        print(f'First Interaction: Block {first_interaction_block} ({format_date(first_interaction_date)})')

    print('\n💰 POSITION SUMMARY')
    print('-' * 80)
    print(f'Current Shares:     {format_units(current_shares, 6)} shares')
    print(f'Current Value:      {format_units(current_value, 6)} USDC')
    print(f'Total Deposited:    {format_units(total_deposited, 6)} USDC')
    print(f'Total Withdrawn:    {format_units(total_withdrawn, 6)} USDC')
    print(f'Net Deposited:      {format_units(net_deposited, 6)} USDC')

    if position.peak_shares > 0:
        print('\n📊 Peak Position:')
        print(f'   Highest Shares:  {format_units(position.peak_shares, 6)} shares')
        if peak_value is not None:
            print(f'   Peak Value:      {format_units(peak_value, 6)} USDC')
        if peak_date:
            print(f'   Peak Date:       Block {position.peak_shares_block} ({format_date(peak_date)})')
        else:
            print(f'   Peak Block:      {position.peak_shares_block}')

        shares_diff = current_shares - position.peak_shares
        shares_diff_pct = (shares_diff * 10000) // position.peak_shares if position.peak_shares else 0
        if shares_diff < 0:
            print(f'   Change from peak: {format_units(-shares_diff, 6)} shares lower ({shares_diff_pct / 100:.2f}%)')
        elif shares_diff == 0:
            print('   Change from peak: Currently at peak!')

    print('\n💰 PRICE PER SHARE ANALYSIS')
    print('-' * 80)
    print(f'Weighted Avg Entry PPS: {format_units(weighted_avg_entry_pps, 6)}')
    print(f'Current PPS:            {format_units(current_pps, 6)}')
    pps_diff = current_pps - weighted_avg_entry_pps
    pps_diff_pct = (pps_diff * 10000) // weighted_avg_entry_pps if weighted_avg_entry_pps else 0
    pps_sign = '+' if pps_diff >= 0 else ''
    print(f'PPS Change:             {pps_sign}{format_units(pps_diff, 6)} ({pps_sign}{pps_diff_pct / 100:.2f}%)')

    print('\n📈 PROFIT/LOSS (Price Per Share Method)')
    print('-' * 80)
    net_profit_sign = '+' if net_profit >= 0 else ''
    gross_profit_sign = '+' if gross_profit >= 0 else ''
    print(f'Gross Profit (before fees): {gross_profit_sign}{format_units(gross_profit, 6)} USDC')
    print(f'Net Profit (after fees):    {net_profit_sign}{format_units(net_profit, 6)} USDC')

    net_profit_pct = (net_profit * 10000) // net_deposited if net_deposited > 0 else 0
    print(f'Return on Investment:       {net_profit_sign}{net_profit_pct / 100:.2f}%')

    print('\n💸 FEES PAID')
    print('-' * 80)
    print(f'Performance Fee Rate:   {performance_fee_bps / 100}%')
    print(f'Total Fees Paid:        {format_units(total_fees, 6)} USDC')
    if gross_profit > 0:
        fee_percentage = (total_fees * 10000) // gross_profit
        print(f'Fees as % of Gross:     {fee_percentage / 100:.2f}%')

    print('\nCalculation Method:')
    print('  • Weighted average entry PPS calculated from deposits and incoming transfers (transfers valued at the block PPS)')
    print('  • Net profit = (Current PPS - Entry PPS) × Current Shares')
    print('  • Gross profit = Net profit / (1 - Fee Rate)')
    print('  • Fees = Gross profit - Net profit')

    print('\n📝 USER EVENTS')
    print('-' * 80)
    print(f'Total Deposits:     {len(deposits)}')
    print(f'Total Withdrawals:  {len(withdrawals)}')
    print(f'Total Transfers:    {len(transfers)} (excluding mint/burn)')
    print(f'  - Transfers IN:   {len(transfers_in)}')
    print(f'  - Transfers OUT:  {len(transfers_out)}')
    print(f'Total Events Processed: {len(position.user_events)}')

    all_events = build_event_timeline(deposits, withdrawals, transfers, depositor_address)
    if all_events:
        print('\n  Events:')
        for index, event in enumerate(all_events, start=1):
            block = event.block_number
            if event.type == 'deposit':
                assets = int(event.data['assets'])
                shares = int(event.data['shares'])
                print(f'    {index}. Block {block}: Deposit {format_units(assets, 6)} USDC → {format_units(shares, 6)} shares')
            elif event.type == 'withdraw':
                assets = int(event.data['assets'])
                shares = int(event.data['shares'])
                print(f'    {index}. Block {block}: Withdraw {format_units(shares, 6)} shares → {format_units(assets, 6)} USDC')
            elif event.type == 'transfer_in':
                value = int(event.data['value'])
                print(f'    {index}. Block {block}: Transfer IN {format_units(value, 6)} shares')
            elif event.type == 'transfer_out':
                value = int(event.data['value'])
                print(f'    {index}. Block {block}: Transfer OUT {format_units(value, 6)} shares')

    print('\n' + '=' * 80)

    plot_balance_profit(position.snapshots, decimals, current_pps, current_shares)



def main() -> None:
    args = sys.argv[1:]
    if not args:
        print('Usage: python3 scripts/calc_depositor_fees.py <depositor-address> [--stable-fees]')
        sys.exit(1)

    # Parse arguments
    depositor_address = None
    check_stable_fees = False

    for arg in args:
        if arg == '--stable-fees':
            check_stable_fees = True
        elif arg.startswith('0x'):
            depositor_address = arg

    if not depositor_address:
        print('Error: Depositor address is required')
        sys.exit(1)
    if len(depositor_address) != 42:
        print('Error: Invalid Ethereum address format')
        sys.exit(1)

    print('Fetching data from Envio indexer...')
    deposits = get_deposit_events(depositor_address)
    withdrawals = get_withdraw_events(depositor_address)
    transfers = get_transfer_events(depositor_address)

    print('Building position timeline...')
    position = calculate_position(
        build_event_timeline(deposits, withdrawals, transfers, depositor_address),
        depositor_address,
    )

    if position.snapshots:
        first_event_block = position.snapshots[0].block_number
        last_event_block = position.snapshots[-1].block_number
    else:
        first_event_block = None
        last_event_block = None

    print('Fetching current vault state...')
    price_per_share_hex = contract_call(VAULT_ADDRESS, PRICE_PER_SHARE_SELECTOR)
    decimals_hex = contract_call(VAULT_ADDRESS, DECIMALS_SELECTOR)
    price_per_share = int(price_per_share_hex, 16)
    decimals = int(decimals_hex, 16)
    current_value = position.current_shares * price_per_share // (10 ** decimals)

    print('Fetching performance fee rate...')
    performance_fee_bps = get_performance_fee_rate(VAULT_ADDRESS)
    if check_stable_fees and first_event_block is not None and last_event_block is not None:
        blocks_to_check = sample_fee_check_blocks(first_event_block, last_event_block)
        print(f'Verifying performance fee stability throughout depositor history ({len(blocks_to_check)} datapoints)...')
        verify_performance_fee_stability(
            VAULT_ADDRESS,
            first_event_block,
            last_event_block,
            performance_fee_bps,
            blocks_to_check=blocks_to_check,
        )
        print(f'Verifying management fee remains zero throughout depositor history ({len(blocks_to_check)} datapoints)...')
        verify_management_fee_zero(VAULT_ADDRESS, blocks_to_check)
    weighted_avg_entry_pps = calculate_weighted_average_entry_pps(position.user_events, decimals)
    profit_and_fees = calculate_incremental_profit_and_fees(
        position.snapshots,
        performance_fee_bps,
        price_per_share,
        position.current_shares,
        decimals,
    )

    all_user_blocks = [
        *map(lambda d: parse_event_id(d.id)[0], deposits),
        *map(lambda w: parse_event_id(w.id)[0], withdrawals),
        *map(lambda t: parse_event_id(t.id)[0], transfers),
    ]
    first_block = min(all_user_blocks) if all_user_blocks else None
    first_date = get_block_timestamp(first_block) if first_block is not None else None

    peak_value = None
    peak_date = None
    if position.peak_shares > 0 and position.peak_shares_block > 0:
        print('Calculating peak position value...')
        try:
            peak_price = get_price_per_share_at_block(position.peak_shares_block)
            peak_value = position.peak_shares * peak_price // (10 ** decimals)
            peak_date = get_block_timestamp(position.peak_shares_block)
        except Exception as exc:
            print(f'Warning: could not fetch peak position value ({exc})')

    format_output(
        depositor_address,
        deposits,
        withdrawals,
        transfers,
        position,
        current_value,
        weighted_avg_entry_pps,
        profit_and_fees,
        performance_fee_bps,
        price_per_share,
        first_date,
        first_block,
        peak_value,
        peak_date,
        decimals,
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}')
        sys.exit(1)
