# Bitcoin UTXO Simulator

A comprehensive C++ educational simulator for understanding Bitcoin's UTXO (Unspent Transaction Output) model, transaction management, mempool operations, blockchain mining, dynamic fee mechanisms, and block reward halving.

## Overview

This project implements a sophisticated Bitcoin transaction simulator that demonstrates how the UTXO model works in cryptocurrency systems. It provides an interactive command-line interface for creating transactions, managing a mempool with dynamic fee-based prioritization, mining blocks with configurable transaction limits, implementing block reward halving, and maintaining a complete blockchain history with visual representation.

## Features

- **UTXO Management**: Track unspent transaction outputs with owner information and balance calculation
- **Transaction Creation**: Create and validate complex transactions with multiple inputs and outputs
- **Dynamic Fee System**: Configurable percentage-based transaction fees with automatic calculation
- **Mempool with Priority Picking**: Fee-based transaction prioritization with size limits and double-spend prevention
- **Block Mining**: Mine blocks with configurable transaction limits and intelligent transaction selection
- **Block Rewards with Halving**: Configurable block rewards with periodic halving mechanism
- **Blockchain Viewing**: Visual ASCII-art blockchain display with complete transaction history
- **Interactive UI**: User-friendly command-line interface for all operations
- **Comprehensive Test Suite**: 13 built-in test scenarios covering all functionality

## System Architecture

The simulator consists of five main components working together:

### 1. UTXO Manager (`utxo_manager.h`)
Manages the global set of unspent transaction outputs.

**Key Functions:**
- `add_utxo()`: Adds a new UTXO to the set
- `remove_utxo()`: Removes a spent UTXO from the set
- `get_balance()`: Calculates total balance for an owner
- `get_utxos_for_owner()`: Retrieves all UTXOs belonging to a specific owner
- `exists()`: Checks if a UTXO exists in the set
- `display()`: Shows all UTXOs in a formatted table

**Data Structure:**
```cpp
map<pair<string, int>, UTXO> utxo_set;
// Key: (transaction_id, output_index)
// Value: UTXO{tx_id, index, amount, owner}
```

### 2. Transaction (`transaction.h`)
Defines the structure of transactions with inputs and outputs.

**Components:**
- **Input**: References a previous transaction output to spend
  - `prev_tx_id`: ID of the transaction containing the UTXO
  - `index`: Output index in that transaction
  - `owner`: Current owner of the UTXO
- **Output**: Creates new UTXOs
  - `amount`: Amount of Bitcoin
  - `address`: Recipient's address
- **Transaction**: Complete transaction structure
  - `tx_id`: Unique transaction identifier
  - `inputs`: Vector of inputs to spend
  - `outputs`: Vector of outputs to create
  - `fee`: Transaction fee (calculated dynamically)

### 3. Mempool (`mempool.h`)
Manages pending transactions before they are mined into blocks.

**Key Features:**
- **Double-Spend Prevention**: Tracks spent UTXOs to prevent conflicts
- **Transaction Validation**: Comprehensive validation before acceptance
- **Priority-Based Selection**: Orders transactions by fee for mining
- **Size Limiting**: Enforces maximum mempool size with eviction
- **Dynamic Fee Calculation**: Fees calculated as `total_input - total_output`

**Key Functions:**
- `add_transaction()`: Validates and adds transaction to mempool
- `remove_transaction()`: Removes a transaction (after mining)
- `get_top_transactions(n)`: Returns top N transactions by fee
- `clear()`: Empties the mempool
- `display()`: Shows all pending transactions

**Validation Checks:**
1. All input UTXOs exist in the UTXO set
2. No double-spending within the same transaction
3. No UTXOs already spent in other mempool transactions
4. Sufficient funds (inputs ≥ outputs)
5. No negative output amounts

### 4. Block (`block.h`)
Represents blockchain blocks with mining functionality.

**Block Structure:**
```cpp
struct Block {
    int block_height;           // Block number in the chain
    string miner;               // Miner's address
    vector<Transaction> transactions;  // Included transactions
    double total_fees;          // Sum of all transaction fees
    double block_reward;        // Block subsidy (with halving)
    string timestamp;           // Block creation time
};
```

**Mining Process (`mine_block` function):**
1. Select top transactions from mempool based on fees
2. Process each transaction:
   - Remove spent input UTXOs
   - Add new output UTXOs
   - Accumulate transaction fees
   - Remove from mempool
3. Create coinbase transaction for miner
4. Award miner: block_reward + total_fees
5. Timestamp the block
6. Return the newly mined block

### 5. Simulator (`simulator.h`)
Main orchestrator providing the interactive interface.

**Configuration Parameters:**
- `gas_fee`: Percentage fee for transactions (e.g., 0.01 = 1%)
- `block_reward`: Initial block reward in BTC
- `mempool_maxsize`: Maximum transactions in mempool
- `k`: Number of blocks between reward halvings

**Key Functions:**
- `run()`: Main interactive loop
- `create_transaction_ui()`: User interface for creating transactions
- `display_blockchain()`: Visual blockchain display with ASCII art
- `run_test_scenarios()`: Executes comprehensive test suite

## Getting Started

### Prerequisites

- C++ compiler with C++11 support (g++, clang++)
- Standard C++ library

### Compilation

```bash
g++ -std=c++11 -o simulator src/main.cpp
```

Or with optimizations:
```bash
g++ -std=c++11 -O2 -o simulator src/main.cpp
```

The `main.cpp` file initializes the simulator with specific parameters:
```cpp
#include "simulator.h"

int main() {
    // Parameters: gas_fee, block_reward, mempool_maxsize, halving_interval
    Simulator sim(0.01, 6.25, 5, 2);
    sim.run();
    return 0;
}
```

**Parameter Explanation:**
- `0.01`: 1% transaction fee
- `6.25`: Initial block reward (current Bitcoin reward as of 2020)
- `5`: Maximum 5 transactions in mempool
- `2`: Block reward halves every 2 blocks (for testing; real Bitcoin is 210,000)
```

### Running the Simulator

```bash
./simulator
```

## Usage

The simulator provides an interactive menu with the following options:

1. **Create new transaction**: Send Bitcoin between users
2. **View UTXO set**: Display all unspent transaction outputs
3. **View mempool**: Show pending transactions with fees
4. **Mine block**: Mine a block with transactions from the mempool
5. **View blockchain**: Display the complete blockchain with visual representation
6. **Run test scenarios**: Execute 13 comprehensive test cases
7. **Exit**: Quit the simulator

### Initial State

The simulator starts with a genesis block containing UTXOs for five users:
- Alice: 50 BTC
- Bob: 30 BTC
- Charlie: 20 BTC
- David: 10 BTC
- Eve: 5 BTC

Total initial supply: 115 BTC

## Detailed Feature Documentation

### 1. Dynamic Fee System

The simulator implements a sophisticated dynamic fee mechanism that closely mirrors real Bitcoin economics.

#### Fee Calculation

**Method 1: Percentage-Based (Used in Transaction Creation UI)**
```
Total Cost = Amount × (1 + gas_fee)
Fee = Total Cost - Amount = Amount × gas_fee
Change = Total Input - Total Cost
```

**Example with 1% fee (gas_fee = 0.01):**
- Alice sends 10 BTC to Bob
- Total cost = 10 × (1 + 0.01) = 10.10 BTC
- Fee = 0.10 BTC
- If Alice has 50 BTC UTXO:
  - Input: 50 BTC
  - Output to Bob: 10 BTC
  - Output to Alice (change): 50 - 10.10 = 39.90 BTC
  - Fee collected by miner: 0.10 BTC

**Method 2: Implicit Fee (Used in Mempool Validation)**
```
Fee = Total Input - Total Output
```

This method is used during mempool validation to calculate the actual fee:
- Sum all input amounts
- Sum all output amounts
- Difference is the transaction fee
- This allows for flexible fee markets

#### Fee Configuration

Fees are configured at simulator initialization:
```cpp
Simulator sim(0.01, 6.25, 5, 2);
         // ^^^^ 1% transaction fee
```

#### Fee Incentives

- Higher fees = higher priority in mempool
- Miners select transactions with highest fees
- Miners receive all transaction fees from mined blocks
- Zero-fee transactions are allowed but have lowest priority

### 2. Mempool Transaction Picking (Priority System)

The mempool implements a sophisticated priority system for transaction selection.

#### Transaction Priority Algorithm

**Sorting Criteria:**
```cpp
sort(transactions.begin(), transactions.end(), 
     [](const Transaction& a, const Transaction& b) {
         return a.fee > b.fee;  // Higher fee = higher priority
     });
```

**Selection Process:**
1. **Validation**: Transaction must pass all validation checks
2. **Priority Calculation**: Based solely on absolute fee amount
3. **Ordering**: Transactions sorted by fee in descending order
4. **Selection**: Top N transactions selected for mining

#### Mempool Management

**Size Limiting:**
- Configurable maximum size (default: 5 transactions)
- When full, lowest-fee transactions are evicted
- Evicted transactions' spent UTXOs are released back to available pool

**Double-Spend Prevention:**
- Tracks all spent UTXOs across mempool transactions
- New transactions cannot spend already-spent UTXOs
- First-seen rule: first transaction wins in conflicts

**Transaction Selection for Mining:**
```cpp
vector<Transaction> get_top_transactions(int n) {
    // Returns top N transactions by fee
    // Used during block mining to select which transactions to include
}
```

**Example Mempool State:**
```
TX ID                     Inputs  Outputs  Fee
tx_alice_bob_123          1       2        0.500
tx_charlie_david_456      1       2        0.300
tx_eve_alice_789          2       1        0.250
tx_bob_charlie_321        1       2        0.100
```
Mining will select transactions in order: 123 → 456 → 789 → 321

### 3. Block Rewards and Halving Mechanism

The simulator implements Bitcoin's block reward system with configurable halving.

#### Block Reward Structure

**Initial Configuration:**
```cpp
Simulator sim(0.01, 6.25, 5, 2);
              //  ^^^^  ^^ 
              //   |     |
              // reward  halving interval (k blocks)
```

**Miner Compensation:**
```
Total Miner Reward = Block Reward + Sum of Transaction Fees
```

**Example:**
- Block reward: 6.25 BTC
- Transaction fees: 0.50 BTC
- Total miner reward: 6.75 BTC

#### Halving Implementation

**Halving Logic:**
```cpp
if (counter % k == 0)
    block_reward /= 2;
```

**Halving Schedule Example (k=2):**
- Blocks 1-2: 6.25 BTC per block
- Blocks 3-4: 3.125 BTC per block
- Blocks 5-6: 1.5625 BTC per block
- Blocks 7-8: 0.78125 BTC per block
- ...continues halving

**Real Bitcoin Comparison:**
- Bitcoin halves every 210,000 blocks (~4 years)
- Current reward: 6.25 BTC (as of 2020)
- Historical rewards:
  - 2009-2012: 50 BTC
  - 2012-2016: 25 BTC
  - 2016-2020: 12.5 BTC
  - 2020-2024: 6.25 BTC
  - 2024+: 3.125 BTC

#### Coinbase Transaction

When a block is mined, a special "coinbase" transaction is created:
```cpp
string coinbase_tx_id = "coinbase_block_" + to_string(block_height);
utxo_manager.add_utxo(coinbase_tx_id, 0, total_miner_reward, miner_address);
```

This UTXO represents the miner's reward and can be spent in future transactions.

### 4. Blockchain Viewing and Visualization

The simulator provides detailed blockchain visualization with ASCII art formatting.

#### Visual Block Display

Each block is displayed as an ASCII box with complete information:

```
+--------------------------------------------------+
| Block #1 | Miner: Alice                          |
| Time: Mon Feb  3 14:56:04 2026                   |
| Reward: 6.250 | Fees: 0.500 BTC                  |
| TXs: 4                                           |
|   +-- a1b2c3d4... [1->2] Fee: 0.200              |
|   +-- e5f6g7h8... [2->1] Fee: 0.150              |
|   +-- i9j0k1l2... [1->2] Fee: 0.100              |
|   +-- m3n4o5p6... [1->1] Fee: 0.050              |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
| Block #2 | Miner: Bob                            |
| Time: Mon Feb  3 14:58:15 2026                   |
| Reward: 3.125 | Fees: 0.300 BTC                  |
| TXs: 3                                           |
|   +-- q7r8s9t0... [1->2] Fee: 0.150              |
|   +-- u1v2w3x4... [1->2] Fee: 0.100              |
|   +-- y5z6a7b8... [2->1] Fee: 0.050              |
+--------------------------------------------------+
```

#### Block Information Display

Each block shows:
- **Block Height**: Sequential block number
- **Miner**: Address that mined the block
- **Timestamp**: Exact time of block creation
- **Reward**: Block subsidy (with halving)
- **Fees**: Total transaction fees collected
- **Transaction Count**: Number of transactions included
- **Transaction Summary**: For each transaction:
  - Transaction ID (first 8 characters)
  - Input count → Output count
  - Transaction fee

#### Blockchain Navigation

The `display_blockchain()` function:
- Shows complete chain from genesis to latest block
- Visual arrows connect blocks sequentially
- Dynamic width adjustment based on content
- Formatted with proper alignment and spacing

**Usage:**
```
Main Menu:
5. View blockchain
Enter choice: 5
```

### 5. Transaction Creation Workflow

Step-by-step process for creating a transaction:

#### User Interaction Flow

1. **Select Sender**
   ```
   Enter sender: Alice
   Available balance: 50.000 BTC
   ```

2. **Check Balance**
   - System automatically displays sender's available balance
   - Balance = sum of all UTXOs owned by sender

3. **Select Recipient**
   ```
   Enter recipient: Bob
   ```

4. **Enter Amount**
   ```
   Enter amount: 10
   ```

5. **Automatic Processing**
   - Select sufficient UTXOs to cover amount + fee
   - Calculate fee based on gas_fee percentage
   - Create output to recipient
   - Create change output back to sender (if needed)
   - Generate unique transaction ID

6. **Validation and Confirmation**
   ```
   Creating transaction...
   Transaction valid! Fee: 0.100
   Transaction ID: tx_Alice_Bob_847
   Transaction added to mempool.
   Mempool now has 1 transactions.
   ```

#### UTXO Selection Algorithm

```cpp
// Select UTXOs until we have enough for amount + fee
double total_input = 0;
for (const auto& utxo : sender_utxos) {
    // Skip UTXOs already in mempool
    if (mempool.spent_utxos.count({utxo.tx_id, utxo.index}))
        continue;
    
    tx.inputs.push_back({utxo.tx_id, utxo.index, utxo.owner});
    total_input += utxo.amount;
    
    // Stop when we have enough
    if (total_input >= amount * (1 + gas_fee))
        break;
}
```

#### Change Calculation

```cpp
double change = total_input - amount * (1 + gas_fee);
if (change > 0) {
    tx.outputs.push_back({change, sender});  // Return change to sender
}
```

### 6. Mining Process

Detailed mining workflow and block creation:

#### Mining Steps

1. **Select Miner**
   ```
   Enter miner name: Alice
   ```

2. **Transaction Selection**
   ```
   Mining block...
   Selected 4 transactions from mempool.
   ```
   - Selects top N transactions by fee (default: 4)
   - Configurable via `num_txs` parameter in `mine_block()`

3. **UTXO Processing**
   - For each transaction in the block:
     - Remove spent input UTXOs from UTXO set
     - Add new output UTXOs to UTXO set
     - Accumulate transaction fees
     - Remove transaction from mempool

4. **Reward Distribution**
   ```
   Block Reward: 6.250 BTC
   Total fees: 0.500 BTC
   Miner Alice receives 6.750 BTC
   Block mined successfully!
   ```

5. **Block Creation**
   - Create coinbase transaction for miner
   - Timestamp the block
   - Add block to blockchain
   - Apply reward halving if applicable

#### Configurable Mining Parameters

**Transaction Limit per Block:**
```cpp
Block new_block = mine_block(miner, mempool, utxo_manager, 
                             blockchain.size() + 1, block_reward, 4);
                             //                                  ^
                             // Max 4 transactions per block
```

**Block Reward Schedule:**
- Controlled by halving interval `k`
- Automatic halving every k blocks
- Miner receives reward + all transaction fees

## Key Concepts Demonstrated

### UTXO Model vs Account Model

**UTXO Model (This Simulator & Bitcoin):**
- Transactions consume entire UTXOs
- Change returned as new UTXO
- Parallel processing possible
- No account balances, only UTXOs

**Account Model (Ethereum):**
- Accounts have balances
- Transactions adjust balances
- Sequential processing required
- Simpler to understand

**Why UTXO?**
- Better privacy (new addresses for change)
- Easier to parallelize
- Simpler to verify
- Clearer ownership tracking

### Transaction Validation

Comprehensive validation ensures blockchain integrity:

**Validation Checklist:**
1. ✓ **Input Existence**: All input UTXOs must exist in UTXO set
   ```
   Error: "Input UTXO does not exist: {tx_id}:{index}"
   ```

2. ✓ **No Double-Spending in Same Transaction**: Each input must be unique
   ```
   Error: "Double-spending in same transaction: {tx_id}:{index}"
   ```

3. ✓ **No Mempool Conflicts**: UTXOs not already spent in pending transactions
   ```
   Error: "UTXO already spent in mempool: {tx_id}:{index}"
   ```

4. ✓ **Sufficient Funds**: Total inputs ≥ Total outputs
   ```
   Error: "Insufficient funds: Input (X) < Output (Y)"
   ```

5. ✓ **Positive Amounts**: No negative output values
   ```
   Error: "Negative output amount"
   ```

### Mempool Management

The mempool acts as a waiting area for unconfirmed transactions:

**Mempool Lifecycle:**
```
Transaction Created
       ↓
   Validation
       ↓
   Add to Mempool (if valid)
       ↓
   Wait for Mining
       ↓
   Included in Block
       ↓
   Removed from Mempool
```

**Key Features:**
- **Priority Queue**: Orders transactions by fee
- **Spent UTXO Tracking**: Prevents double-spending across mempool
- **Size Management**: Evicts low-fee transactions when full
- **First-Seen Rule**: First transaction spending a UTXO wins

**Mempool States:**
```
Empty: Ready for new transactions
Partial: Has space for more transactions
Full: At capacity, eviction may occur
```

### Block Mining

Mining is the process of adding transactions to the blockchain:

**Mining Algorithm:**
```
1. SELECT top transactions by fee from mempool
2. FOR EACH transaction:
     - Validate inputs exist
     - Remove spent UTXOs
     - Add new output UTXOs
     - Accumulate fees
3. CREATE coinbase transaction
4. AWARD miner: block_reward + total_fees
5. TIMESTAMP block
6. ADD block to blockchain
7. UPDATE block_reward (apply halving if needed)
```

**Miner Incentives:**
- Block reward (subsidy)
- Transaction fees
- Total = block_reward + Σ(transaction_fees)

**Block Finality:**
Once a block is mined:
- All transactions are confirmed
- UTXO set is updated permanently
- Transactions removed from mempool
- Block added to immutable chain

### Double-Spending Prevention

Multiple layers protect against double-spending:

**Layer 1: Transaction-Level**
- Cannot use same UTXO multiple times in one transaction
- Detected during transaction creation

**Layer 2: Mempool-Level**
- Tracks spent UTXOs across all pending transactions
- First transaction to spend a UTXO locks it
- Subsequent attempts rejected

**Layer 3: Blockchain-Level**
- Once UTXO is spent in a mined block, it's removed from UTXO set
- Cannot be spent again in future transactions
- Permanent and immutable

**Race Attack Simulation:**
```
Time    Event
----    -----
T0:     Alice sends UTXO to Merchant (low fee)
T1:     Alice sends SAME UTXO to herself (high fee)
T2:     Mempool accepts first transaction (T0)
T3:     Mempool rejects second transaction (T1)
T4:     Miner includes first transaction in block
T5:     UTXO removed from UTXO set
```

### Fee Market and Transaction Economics

**Fee Calculation Methods:**

1. **Percentage-Based (User-Facing):**
   ```
   Fee = Amount × gas_fee_percentage
   Total Cost = Amount + Fee
   ```

2. **Implicit (Validation):**
   ```
   Fee = Total Inputs - Total Outputs
   ```

**Fee Dynamics:**
- Higher fees = faster confirmation
- Zero-fee transactions allowed (lowest priority)
- Miners maximize revenue by selecting high-fee transactions
- Fee market emerges naturally from supply/demand

**Transaction Priority Formula:**
```
Priority = Fee (absolute amount in BTC)
```

Simple but effective: transactions sorted by fee in descending order.

## Test Suite

The simulator includes 13 comprehensive test scenarios:

### Test 1: Basic Valid Transaction
- **Purpose**: Verify basic transaction creation and fee calculation
- **Scenario**: Alice (50 BTC) sends 10 BTC to Bob
- **Validates**: Change calculation, fee computation, output creation

### Test 2: Multiple Inputs
- **Purpose**: Test transaction with multiple input UTXOs
- **Scenario**: Alice uses two UTXOs (50 + 20 BTC) to send 60 BTC to Bob
- **Validates**: Input aggregation, total fee calculation

### Test 3: Double-Spend in Same Transaction
- **Purpose**: Ensure same UTXO cannot be used twice in one transaction
- **Scenario**: Transaction tries to spend same UTXO as two inputs
- **Expected**: Rejection with clear error message

### Test 4: Mempool Double-Spend
- **Purpose**: Prevent spending UTXO already in mempool
- **Scenario**: Two transactions try to spend same UTXO
- **Expected**: First accepted, second rejected (first-seen rule)

### Test 5: Insufficient Funds
- **Purpose**: Validate input amount checking
- **Scenario**: Bob (30 BTC) tries to send 35 BTC
- **Expected**: Rejection due to insufficient funds

### Test 6: Negative Amount
- **Purpose**: Prevent negative output amounts
- **Scenario**: Transaction with negative output value
- **Expected**: Immediate rejection

### Test 7: Zero Fee Transaction
- **Purpose**: Allow transactions with no fee
- **Scenario**: Inputs exactly equal outputs (fee = 0)
- **Expected**: Accepted but lowest priority

### Test 8: Race Attack Simulation
- **Purpose**: Demonstrate first-seen rule protection
- **Scenario**: Low-fee transaction arrives first, high-fee conflict arrives second
- **Expected**: First transaction wins regardless of fee

### Test 9: Complete Mining Flow
- **Purpose**: Validate entire mining process
- **Scenario**: Create transactions, mine block, verify UTXO updates
- **Validates**: UTXO updates, miner rewards, mempool clearing

### Test 10: Unconfirmed Chain
- **Purpose**: Prevent spending unconfirmed outputs
- **Scenario**: Bob tries to spend output from unmined transaction
- **Expected**: Rejection - UTXO doesn't exist yet

### Test 11: Mempool Transaction Limit
- **Purpose**: Test mempool size enforcement
- **Scenario**: Add transactions until mempool is full
- **Expected**: Rejection with "Mempool is full" error, eviction of low-fee transactions

### Test 12: Block Transaction Limit
- **Purpose**: Verify block size constraints
- **Scenario**: Mine block with more transactions available than limit
- **Expected**: Only specified number included, rest remain in mempool

### Test 13: Block Reward Halving
- **Purpose**: Test reward halving mechanism
- **Scenario**: Mine k blocks and verify reward halves
- **Validates**: Correct reward at each epoch, proper halving schedule

**Running Tests:**
```
Main Menu:
6. Run test scenarios
Enter choice: 6
```

**Test Output:**
```
========== Running Test Suite ==========
[PASS] Test 1: Basic Valid Transaction
[PASS] Test 2: Multiple Inputs
[PASS] Test 3: Double-Spend in Same Transaction
...
========================================
Passed 13 / 13 tests.
```

## Complete Usage Examples

### Example 1: Simple Transaction

```
=== Bitcoin Transaction Simulator ===
Initial UTXOs (Genesis Block):
TX ID                Index     Amount         Owner
genesis              0         50.000         Alice
genesis              1         30.000         Bob
genesis              2         20.000         Charlie

Main Menu:
1. Create new transaction
Enter choice: 1

Enter sender: Alice
Available balance: 50.000 BTC
Enter recipient: Bob
Enter amount: 10

Creating transaction...
Transaction valid! Fee: 0.100
Transaction ID: tx_Alice_Bob_123
Transaction added to mempool.
Mempool now has 1 transactions.
```

### Example 2: View Mempool

```
Main Menu:
3. View mempool
Enter choice: 3

--- Current Mempool ---
TX ID                     Inputs  Outputs  Fee
tx_Alice_Bob_123          1       2        0.100
```

### Example 3: Mine Block

```
Main Menu:
4. Mine block
Enter choice: 4

Enter miner name: Charlie
Mining block...
Selected 1 transactions from mempool.
Block Reward: 6.250 BTC
Total fees: 0.100 BTC
Miner Charlie receives 6.350 BTC
Block mined successfully!
```

### Example 4: View Blockchain

```
Main Menu:
5. View blockchain
Enter choice: 5

+--------------------------------------------------+
| Block #1 | Miner: Charlie                        |
| Time: Mon Feb  3 14:56:04 2026                   |
| Reward: 6.250 | Fees: 0.100 BTC                  |
| TXs: 1                                           |
|   +-- tx_Alice_... [1->2] Fee: 0.100            |
+--------------------------------------------------+
```

### Example 5: Check Updated UTXO Set

```
Main Menu:
2. View UTXO set
Enter choice: 2

--- Current UTXO Set ---
TX ID                Index     Amount         Owner
genesis              1         30.000         Bob
genesis              2         20.000         Charlie
genesis              3         10.000         David
genesis              4         5.000          Eve
tx_Alice_Bob_123     0         10.000         Bob
tx_Alice_Bob_123     1         39.900         Alice
coinbase_block_1     0         6.350          Charlie
```

## File Structure

```
CS216-nchainai-UTXO-Simulator/
├── src/
│   ├── utxo_manager.h    # UTXO set management
│   │   ├── UTXO struct
│   │   ├── UTXOManager class
│   │   └── Balance calculation and UTXO tracking
│   ├── transaction.h     # Transaction data structures
│   │   ├── Input struct (references previous outputs)
│   │   ├── Output struct (creates new UTXOs)
│   │   └── Transaction struct (complete tx definition)
│   ├── mempool.h        # Transaction pool management
│   │   ├── Mempool class
│   │   ├── Transaction validation logic
│   │   ├── Priority-based selection
│   │   └── Double-spend prevention
│   ├── block.h          # Block structure and mining
│   │   ├── Block struct
│   │   ├── mine_block() function
│   │   └── Block display with ASCII art
│   ├── simulator.h      # Main simulator with UI
│   │   ├── Simulator class
│   │   ├── Interactive menu system
│   │   ├── Transaction creation UI
│   │   ├── Blockchain visualization
│   │   └── Test scenario runner
│   └── main.cpp         # Entry point
│       └── Simulator initialization with parameters
├── test/
│   └── tests.h         # Comprehensive test suite
│       └── 13 test scenarios covering all functionality
├── Assignment 1.pdf    # Project assignment document
└── README.md          # This comprehensive documentation
```

## Advanced Features

### 1. Change Management

**Automatic Change Calculation:**
When a transaction uses UTXOs worth more than the payment amount, the simulator automatically creates a change output back to the sender.

**Example:**
```
Alice has: 50 BTC UTXO
Alice sends: 10 BTC to Bob
Fee: 1% = 0.10 BTC

Transaction structure:
  Inputs:  [50 BTC from Alice]
  Outputs: [10 BTC to Bob, 39.90 BTC to Alice (change)]
  Fee:     0.10 BTC
```

### 2. UTXO Consolidation

When a sender has multiple small UTXOs, the simulator intelligently selects and combines them:

**Example:**
```
Alice has:
  - UTXO1: 5 BTC
  - UTXO2: 3 BTC
  - UTXO3: 4 BTC

Alice sends 10 BTC to Bob:
  Inputs:  [5 BTC, 3 BTC, 4 BTC] = 12 BTC total
  Outputs: [10 BTC to Bob, 1.90 BTC to Alice (change)]
  Fee:     0.10 BTC (1% of 10 BTC)
```

### 3. Race Attack Protection

The mempool implements first-seen rule to prevent race attacks:

**Scenario:**
```
T0: Alice broadcasts: Send 50 BTC to Merchant (fee: 0.10)
T1: Merchant sees transaction, delivers goods
T2: Alice broadcasts: Send 50 BTC to herself (fee: 5.00)
T3: Mempool rejects second transaction (UTXO already spent)
T4: First transaction gets mined
T5: Merchant's transaction is confirmed
```

### 4. Fee Market Dynamics

**Low Fee Consequences:**
- Lower priority in mempool
- May be evicted if mempool fills
- Slower confirmation time

**High Fee Benefits:**
- Highest priority in mempool
- Selected first for mining
- Faster confirmation time

**Example Fee Competition:**
```
Mempool State:
1. tx_A: 0.500 BTC fee ← Will be mined first
2. tx_B: 0.300 BTC fee ← Will be mined second
3. tx_C: 0.250 BTC fee ← Will be mined third
4. tx_D: 0.100 BTC fee ← Will be mined fourth
5. tx_E: 0.050 BTC fee ← May be evicted if mempool fills
```

### 5. Block Reward Economics

**Inflation Control:**
Block rewards decrease over time through halving:

```
Block Range    Reward    New Supply per Block
-----------    ------    -------------------
1-2            6.25      6.25 BTC
3-4            3.125     3.125 BTC
5-6            1.5625    1.5625 BTC
...            ...       ...
Eventually     0.000     0 BTC (fees only)
```

**Total Supply Calculation:**
With halving every k blocks and initial reward R:
```
Total Supply = R × k × (1 + 1/2 + 1/4 + 1/8 + ...)
             = R × k × 2
             
Example (k=2, R=6.25):
Total Supply = 6.25 × 2 × 2 = 25 BTC (theoretical max)
```

### 6. Transaction Lifecycle

**Complete Transaction Journey:**
```
1. CREATION
   ├─ User specifies sender, recipient, amount
   ├─ System selects UTXOs
   ├─ Calculates fee and change
   └─ Generates transaction ID

2. VALIDATION
   ├─ Check input UTXOs exist
   ├─ Check for double-spending
   ├─ Verify sufficient funds
   └─ Validate output amounts

3. MEMPOOL
   ├─ Add to pending transactions
   ├─ Mark UTXOs as spent in mempool
   ├─ Calculate priority (based on fee)
   └─ Wait for mining

4. MINING
   ├─ Selected by miner (highest fee first)
   ├─ Included in block
   ├─ UTXOs processed (inputs removed, outputs added)
   └─ Transaction fee collected by miner

5. CONFIRMATION
   ├─ Block added to blockchain
   ├─ Transaction permanently recorded
   ├─ UTXOs finalized
   └─ Cannot be reversed
```

## Configuration Guide

### Simulator Parameters

**Full Constructor:**
```cpp
Simulator(double gas_fee, double block_reward, 
          int mempool_maxsize, int halving_interval)
```

**Parameter Details:**

| Parameter | Type | Description | Example | Typical Range |
|-----------|------|-------------|---------|---------------|
| `gas_fee` | double | Transaction fee percentage | 0.01 (1%) | 0.001 - 0.05 |
| `block_reward` | double | Initial block reward in BTC | 6.25 | 1.0 - 50.0 |
| `mempool_maxsize` | int | Max transactions in mempool | 5 | 5 - 100 |
| `halving_interval` | int | Blocks between halvings | 2 | 2 - 210000 |

**Configuration Examples:**

**Test Configuration (Fast Halving):**
```cpp
Simulator sim(0.01, 6.25, 5, 2);
// Quick halving for testing (every 2 blocks)
```

**Bitcoin-like Configuration:**
```cpp
Simulator sim(0.001, 6.25, 100, 210000);
// Simulates real Bitcoin parameters
```

**High Fee Environment:**
```cpp
Simulator sim(0.05, 3.125, 50, 100);
// 5% fees, lower reward, larger mempool
```

### Customizing Genesis Block

To modify initial balances, edit `simulator.h`:
```cpp
Simulator(/* parameters */) {
    // Customize initial distribution
    utxo_manager.add_utxo("genesis", 0, 100.0, "Alice");
    utxo_manager.add_utxo("genesis", 1, 50.0, "Bob");
    utxo_manager.add_utxo("genesis", 2, 25.0, "Charlie");
    // Add more users as needed
}
```

### Adjusting Block Size

Modify the mining transaction limit in `simulator.h`:
```cpp
Block new_block = mine_block(miner, mempool, utxo_manager, 
                             blockchain.size() + 1, block_reward, 10);
                             //                                  ^^
                             // Change this to adjust block capacity
```

## Educational Value

This simulator helps understand:

### Core Bitcoin Concepts
- **UTXO Model**: How Bitcoin tracks ownership through unspent outputs
- **Transaction Structure**: Inputs, outputs, and the transformation process
- **Change Mechanism**: Why and how change is created
- **Fee Markets**: Economic incentives for transaction inclusion

### Blockchain Mechanics
- **Block Structure**: What data blocks contain
- **Block Mining**: How transactions move from mempool to blockchain
- **Blockchain Growth**: Sequential addition of blocks
- **Immutability**: Why confirmed transactions cannot be reversed

### Economic Incentives
- **Miner Rewards**: Block subsidies + transaction fees
- **Fee Competition**: How users compete for block space
- **Reward Halving**: Bitcoin's controlled supply mechanism
- **Economic Security**: Why miners are incentivized to follow rules

### Security Mechanisms
- **Double-Spend Prevention**: Multiple layers of protection
- **Transaction Validation**: Comprehensive security checks
- **Mempool Management**: Preventing conflicts before mining
- **UTXO Set Integrity**: Maintaining accurate ownership records

### Computer Science Concepts
- **Data Structures**: Maps, vectors, sets for efficient operations
- **Algorithms**: Sorting, searching, validation logic
- **State Management**: Tracking mutable (mempool) and immutable (blockchain) state
- **Event-Driven Systems**: Transaction lifecycle from creation to confirmation


x spending conditions
- ❌ No multisignature transactions
- ❌ No timelocks
- ✓ Simple pay-to-address model

### Block Simplifications
- ❌ No Merkle trees
- ❌ No block headers separate from data
- ❌ Fixed block size (transaction count, not bytes)
- ✓ Simplified block structure

### What IS Realistic
- ✓ UTXO model and lifecycle
- ✓ Transaction structure (inputs/outputs)
- ✓ Fee-based transaction priority
- ✓ Mempool management
- ✓ Double-spend prevention logic
- ✓ Block reward halving mechanism
- ✓ Coinbase transactions
- ✓ Change calculation
- ✓ Balance derived from UTXOs

## Troubleshooting

### Common Issues

**Issue: "Insufficient funds for fee"**
```
Problem: Total UTXOs < Amount × (1 + gas_fee)
Solution: Check sender's balance includes enough for fee
Example: Sending 10 BTC with 1% fee requires 10.10 BTC total
```

**Issue: "UTXO already spent in mempool"**
```
Problem: Trying to spend UTXO that's in a pending transaction
Solution: Wait for pending transaction to be mined, or use different UTXO
```

**Issue: "Mempool is full"**
```
Problem: Mempool at capacity, transaction has low fee
Solution: Increase transaction fee or wait for block to be mined
Current: Low-fee transactions automatically evicted
```

**Issue: "All available UTXOs are already pending in mempool"**
```
Problem: All sender's UTXOs are locked in mempool transactions
Solution: Wait for transactions to be mined to free up UTXOs
```

**Issue: "No transactions to mine"**
```
Problem: Mempool is empty
Solution: Create transactions before attempting to mine
```

### Debugging Tips

**Check UTXO State:**
```
Main Menu → 2. View UTXO set
Verify UTXOs exist and amounts are correct
```

**Check Mempool State:**
```
Main Menu → 3. View mempool
See pending transactions and their fees
```

**View Transaction History:**
```
Main Menu → 5. View blockchain
Review all confirmed transactions
```

**Run Tests for Validation:**
```
Main Menu → 6. Run test scenarios
Verify system is working correctly
```

## Performance Considerations

### Computational Complexity

**UTXO Lookup:** O(log n) - Map-based storage
**Balance Calculation:** O(n) - Iterates all UTXOs
**Transaction Validation:** O(m) - Checks m inputs
**Mempool Sorting:** O(n log n) - Sorts by fee
**Block Mining:** O(k) - Processes k transactions

### Memory Usage

**UTXO Set:** Grows with each transaction output
**Mempool:** Limited by max_size parameter
**Blockchain:** Grows linearly with blocks
**Optimization:** Old blocks could be archived in production system

### Scalability Notes

Current design works well for:
- ✓ Educational demonstrations
- ✓ Small-scale simulations (<1000 transactions)
- ✓ Testing and validation

Would need optimization for:
- ❌ Large-scale simulations (>10,000 transactions)
- ❌ High-frequency transaction creation
- ❌ Production blockchain systems

## Extensions and Improvements

Potential enhancements for advanced learning:

### Beginner Extensions
1. **Multiple Recipients**: Support transactions with multiple different recipients
2. **Transaction History**: Show per-user transaction history
3. **Balance Tracking**: Real-time balance updates
4. **Fee Estimation**: Suggest optimal fees based on mempool state

### Intermediate Extensions
1. **Transaction Priority Algorithms**: Implement fee-per-byte instead of absolute fee
2. **Mempool Replacement**: Replace-by-fee (RBF) functionality
3. **Block Weight Limits**: Limit blocks by total transaction size, not count
4. **UTXO Aging**: Track UTXO age for advanced selection strategies

### Advanced Extensions
1. **Basic Signatures**: Add simple digital signature verification
2. **Multi-Signature**: Require multiple signatures for spending
3. **Timelocks**: Transactions that can only be spent after certain time
4. **Script System**: Basic smart contract functionality
5. **Merkle Trees**: Efficient transaction verification
6. **SPV Clients**: Lightweight verification
7. **Network Simulation**: Multiple nodes with propagation delays
8. **Fork Handling**: Chain reorganization logic

### Research Extensions
1. **Lightning Network**: Off-chain payment channels
2. **Segregated Witness**: Transaction malleability fixes
3. **Confidential Transactions**: Privacy-preserving amounts
4. **Schnorr Signatures**: Signature aggregation
5. **Taproot**: Advanced scripting privacy

## Contributing

This is an educational project for CS216. Contributions should maintain:
- Code clarity and readability
- Educational value
- Comprehensive comments
- Test coverage
- Documentation updates

**Contribution Guidelines:**
1. Fork the repository
2. Create a feature branch
3. Make changes with clear commits
4. Add tests for new features
5. Update documentation
6. Submit pull request with detailed description

## License

This project is created for educational purposes as part of CS216 coursework.

Educational use permitted. Please credit CS216 and nChain AI when using or adapting this code.

## Authors

Created as part of the CS216 - nChain AI UTXO Simulator assignment.

**Course:** CS216 - Blockchain Technology
**Institution:** [Your Institution]
**Assignment:** UTXO Simulator Implementation

## Acknowledgments

- **Satoshi Nakamoto**: Bitcoin whitepaper and original design
- **CS216 Course Materials**: Assignment specifications and guidance
- **nChain AI**: Assignment design and educational framework
- **Bitcoin Core**: Implementation reference and inspiration

## Additional Resources

### Learn More About Bitcoin

**Essential Reading:**
- Bitcoin Whitepaper: "Bitcoin: A Peer-to-Peer Electronic Cash System"
- Mastering Bitcoin by Andreas Antonopoulos
- Bitcoin Developer Documentation

**Technical References:**
- Bitcoin Improvement Proposals (BIPs)
- Bitcoin Core source code
- Bitcoin Stack Exchange

**Online Courses:**
- Coursera: Bitcoin and Cryptocurrency Technologies
- MIT OpenCourseWare: Blockchain and Money
- Bitcoin.org Developer Guide

### Related Projects

- **Bitcoin Core**: Reference implementation
- **btcd**: Alternative Go implementation
- **bitcoinj**: Java library for Bitcoin
- **NBitcoin**: .NET Bitcoin library

### Tools for Further Exploration

- **Bitcoin Testnet**: Practice with real Bitcoin software
- **Blockchain Explorers**: Visualize real transactions
- **Bitcoin Improvement Proposals**: Learn about protocol upgrades
- **Bitcoin Stack Exchange**: Community Q&A

## FAQ

**Q: Why does Bitcoin use the UTXO model instead of accounts?**
A: UTXOs enable parallel processing, better privacy, and simpler verification. Account models require sequential processing and more complex state management.

**Q: What happens to transaction fees?**
A: All transaction fees from a block go to the miner who mines that block, in addition to the block reward.

**Q: Why do rewards halve?**
A: Halving controls inflation and creates scarcity, ensuring Bitcoin approaches a maximum supply (21 million BTC in real Bitcoin).

**Q: Can I send a transaction with zero fee?**
A: Yes, but it has lowest priority and may never be mined if the mempool stays full with higher-fee transactions.

**Q: What is change and why is it created?**
A: UTXOs must be spent entirely. If a UTXO is larger than the payment amount, the excess is returned as "change" in a new UTXO back to the sender.

**Q: Why can't I spend an unconfirmed UTXO?**
A: This simulator requires UTXOs to be confirmed (mined in a block) before spending. This prevents complications from transaction chains and keeps the model simple.

**Q: How are transaction IDs generated?**
A: In this simulator, simple string concatenation. In real Bitcoin, the transaction ID is the double SHA-256 hash of the transaction data.

**Q: What prevents double-spending?**
A: Three layers: (1) validation rejects duplicate inputs in same transaction, (2) mempool tracks spent UTXOs across pending transactions, (3) blockchain permanently removes spent UTXOs.

**Q: Why do some transactions get evicted from the mempool?**
A: When the mempool reaches capacity, the lowest-fee transactions are evicted to make room for new transactions.

**Q: How do miners decide which transactions to include?**
A: Miners maximize profit by selecting transactions with the highest fees, up to the block size limit.

## Complete Workflow Example

Here's a comprehensive walkthrough demonstrating all major features:

### Step 1: Start Simulator
```bash
$ ./simulator

=== Bitcoin Transaction Simulator ===
Initial UTXOs (Genesis Block):
TX ID                Index     Amount         Owner
genesis              0         50.000         Alice
genesis              1         30.000         Bob
genesis              2         20.000         Charlie
genesis              3         10.000         David
genesis              4         5.000          Eve
```

### Step 2: Create Transaction (Alice → Bob)
```
Main Menu:
1. Create new transaction
Enter choice: 1

Enter sender: Alice
Available balance: 50.000 BTC
Enter recipient: Bob
Enter amount: 15

Creating transaction...
Transaction valid! Fee: 0.150
Transaction ID: tx_Alice_Bob_847
Transaction added to mempool.
Mempool now has 1 transactions.
```

### Step 3: Create Transaction (Charlie → David)
```
Enter choice: 1

Enter sender: Charlie
Available balance: 20.000 BTC
Enter recipient: David
Enter amount: 5

Creating transaction...
Transaction valid! Fee: 0.050
Transaction ID: tx_Charlie_David_923
Transaction added to mempool.
Mempool now has 2 transactions.
```

### Step 4: View Mempool (Check Priority)
```
Enter choice: 3

--- Current Mempool ---
TX ID                     Inputs  Outputs  Fee
tx_Alice_Bob_847          1       2        0.150
tx_Charlie_David_923      1       2        0.050
```

**Note:** Alice's transaction has higher fee (0.150 > 0.050), so it will be mined first.

### Step 5: Mine Block
```
Enter choice: 4

Enter miner name: Eve
Mining block...
Selected 2 transactions from mempool.
Block Reward: 6.250 BTC
Total fees: 0.200 BTC
Miner Eve receives 6.450 BTC
Block mined successfully!
```

### Step 6: View Blockchain
```
Enter choice: 5

+--------------------------------------------------+
| Block #1 | Miner: Eve                            |
| Time: Mon Feb  3 14:56:04 2026                   |
| Reward: 6.250 | Fees: 0.200 BTC                  |
| TXs: 2                                           |
|   +-- tx_Alice_... [1->2] Fee: 0.150            |
|   +-- tx_Charl... [1->2] Fee: 0.050             |
+--------------------------------------------------+
```

### Step 7: View Updated UTXO Set
```
Enter choice: 2

--- Current UTXO Set ---
TX ID                Index     Amount         Owner
genesis              1         30.000         Bob
genesis              3         10.000         David
genesis              4         5.000          Eve
tx_Alice_Bob_847     0         15.000         Bob
tx_Alice_Bob_847     1         34.850         Alice
tx_Charlie_David_923 0         5.000          David
tx_Charlie_David_923 1         14.950         Charlie
coinbase_block_1     0         6.450          Eve
```

**Key Changes:**
- ✓ Alice's genesis UTXO (50 BTC) spent → replaced with change (34.850 BTC)
- ✓ Bob received 15 BTC from Alice
- ✓ Charlie's genesis UTXO (20 BTC) spent → replaced with change (14.950 BTC)
- ✓ David received 5 BTC from Charlie
- ✓ Eve received block reward + fees (6.450 BTC)

### Step 8: Verify Balances
```
Alice: 34.850 BTC (was 50, sent 15 + 0.15 fee = 34.85)
Bob: 30 + 15 = 45 BTC (original + received)
Charlie: 14.950 BTC (was 20, sent 5 + 0.05 fee = 14.95)
David: 10 + 5 = 15 BTC (original + received)
Eve: 5 + 6.45 = 11.45 BTC (original + mining reward)
```

### Step 9: Mine Another Block (Reward Halving)
```
Enter choice: 1
[Create another transaction...]

Enter choice: 4
Enter miner name: Bob
Mining block...
Selected 1 transactions from mempool.
Block Reward: 3.125 BTC  ← Notice: Halved! (k=2, every 2 blocks)
Total fees: 0.100 BTC
Miner Bob receives 3.225 BTC
Block mined successfully!
```

### Step 10: View Complete Blockchain
```
Enter choice: 5

+--------------------------------------------------+
| Block #1 | Miner: Eve                            |
| Time: Mon Feb  3 14:56:04 2026                   |
| Reward: 6.250 | Fees: 0.200 BTC                  |
| TXs: 2                                           |
|   +-- tx_Alice_... [1->2] Fee: 0.150            |
|   +-- tx_Charl... [1->2] Fee: 0.050             |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
| Block #2 | Miner: Bob                            |
| Time: Mon Feb  3 14:58:15 2026                   |
| Reward: 3.125 | Fees: 0.100 BTC                  |
| TXs: 1                                           |
|   +-- tx_Eve_A... [1->2] Fee: 0.100             |
+--------------------------------------------------+
```

### Step 11: Run Test Suite
```
Enter choice: 6

========== Running Test Suite ==========

=======================
Running Test 1: Basic Valid Transaction
[PASS] Transaction should be accepted by mempool
[PASS] Transaction is present in mempool
[PASS] Fee should be approx 0.100000 BTC
[PASS] Change output back to Alice present and correct

=======================
Running Test 2: Multiple Inputs
[PASS] Transaction with multiple inputs should be accepted
[PASS] Transaction stored in mempool
[PASS] Fee calculated correctly for multiple inputs

[... 11 more tests ...]

========================================
Passed 13 / 13 tests.
```

### Summary of Demonstrated Features

✅ **Dynamic Fees**: Calculated automatically (1% of amount)
✅ **Mempool Priority**: Higher fee transactions selected first
✅ **Block Rewards**: Miner receives base reward + all fees
✅ **Reward Halving**: Automatic halving every k blocks
✅ **Blockchain Viewing**: Visual ASCII representation
✅ **UTXO Model**: UTXOs created and destroyed properly
✅ **Change Calculation**: Automatic change back to sender
✅ **Balance Tracking**: Derived from UTXO set
✅ **Transaction Validation**: Comprehensive security checks
✅ **Mempool Management**: Priority-based transaction selection

---

**Last Updated:** February 2026
**Version:** 1.0
**Status:** Complete Educational Implementation
