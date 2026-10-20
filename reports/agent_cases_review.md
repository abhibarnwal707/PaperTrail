# Hand check of the 30 case summaries

For each claim: is it true, and does the cited output support it? Mark the box.

## `50528_812D0C600`  truth: RANDOM; SCATTER-GATHER; BIPARTITE

Predicted **FAN-IN**, action **escalate_SAR**, verified True, drafts 1.

> The account has a blend risk score of 5.31 and five different senders in its neighbourhood, indicating a fan‑in pattern. Two‑step cycles are also present, and the highest‑risk payment identified was 14744.79 SAR via ACH. The latest transaction was a 66793.17 SAR ACH send that was not flagged. These factors together suggest notable inbound activity risk.

- [ ] The blend model assigned a risk score of 5.31 to the account  (`alert:50528_812D0C600`)
- [ ] The subgraph shows 5 distinct senders paying the account  (`subgraph:50528_812D0C600`)
- [ ] There are 2 two‑step cycles involving the account  (`subgraph:50528_812D0C600`)
- [ ] The riskiest payment amount identified by SHAP is 14744.79 Saudi Riyal  (`shap:50528_812D0C600`)
- [ ] The most recent payment was a sent ACH transfer of 66793.17 Saudi Riyal and was not flagged by blend  (`history:50528_812D0C600`)

## `148350_812D0C3C0`  truth: BIPARTITE; SCATTER-GATHER; RANDOM

Predicted **FAN-IN**, action **escalate_SAR**, verified True, drafts 1.

> The account shows a strong fan‑in pattern, receiving funds from seven different senders while only sending to two receivers. Two‑step cycles are present, and the blend model flagged four payments, including the highest‑risk ACH receipt of 14744.79 SAR. The blend risk score of 5.31 places the account in the top 50 ranks, indicating significant concern.

- [ ] The account received payments from 7 distinct senders  (`subgraph:148350_812D0C3C0`)
- [ ] The account sent payments to only 2 distinct receivers  (`subgraph:148350_812D0C3C0`)
- [ ] There are 2 two‑step cycles involving the account  (`subgraph:148350_812D0C3C0`)
- [ ] The blend model flagged 4 payments for this account  (`alert:148350_812D0C3C0`)
- [ ] The riskiest payment was an ACH receipt of 14744.79 Saudi Riyal flagged by blend with a score of 5.31  (`history:148350_812D0C3C0`)

## `119_812A09CF0`  truth: STACK; SCATTER-GATHER; BIPARTITE; CYCLE; RANDOM

Predicted **CYCLE**, action **enhanced_review**, verified True, drafts 1.

> Account 119_812A09CF0 exhibits a cyclic pattern with one two-step cycle detected in its network. The account primarily interacts with a single counterparty, 49365_812A09D40, with 20 payments recorded in the history. The riskiest payment involved a reciprocal payment structure where the receiver had previously paid the sender, contributing to a high risk score of 5.06.

- [ ] The account has a risk score of 5.06 and a rank of 70 in the blend model.  (`alert:119_812A09CF0`)
- [ ] The network subgraph shows 1 two-step cycle involving the account.  (`subgraph:119_812A09CF0`)
- [ ] The account has 2 distinct receivers and 1 distinct sender in its direct payment history.  (`subgraph:119_812A09CF0`)
- [ ] The most recent payment in the history is a sent ACH transfer of 54541.45 Saudi Riyal to 49365_812A09D40, which was not flagged.  (`history:119_812A09CF0`)
- [ ] The riskiest payment was an ACH transfer where the receiver had already paid the sender before, raising the risk score.  (`shap:119_812A09CF0`)
- [ ] The GNN model flagged 6 payments for this account, ranking it 5th in the queue.  (`alert:119_812A09CF0`)

## `1467_800728800`  truth: CYCLE

Predicted **FAN-OUT**, action **escalate_SAR**, verified True, drafts 1.

> The account has a high fan‑out with 5 receivers and a modest fan‑in of 2 senders. Two payments in its neighbourhood are flagged, and the account itself is involved in one flagged payment. The blend risk score of 5.22 places it in the top 50 of the queue. This pattern is consistent with a FAN‑OUT typology.

- [ ] the account sent payments to 5 distinct receivers  (`subgraph:1467_800728800`)
- [ ] the account received payments from 2 distinct senders  (`subgraph:1467_800728800`)
- [ ] there are 2 flagged payments in the neighbourhood of the account  (`subgraph:1467_800728800`)
- [ ] the account itself touches 1 flagged payment  (`subgraph:1467_800728800`)
- [ ] the neighbourhood shows 114 payments, 71 sent and 29 received  (`subgraph:1467_800728800`)

## `17854_811AF2F80`  truth: CYCLE

Predicted **FAN-IN**, action **enhanced_review**, verified True, drafts 1.

> The account has a moderate risk score of 5.22 and shows a fan‑in pattern with two senders and one receiver. Two of the three models flagged payments, and the riskiest payment is a $14655.44 ACH transfer. The account also received a $12715.37 ACH payment that was flagged. This activity warrants further scrutiny.

- [ ] The account sent 3 payments and received 5 payments in its neighbourhood  (`subgraph:17854_811AF2F80`)
- [ ] The account has 2 distinct senders and 1 distinct receiver  (`subgraph:17854_811AF2F80`)
- [ ] The riskiest payment is a $14655.44 ACH transfer flagged by blend  (`shap:17854_811AF2F80`)
- [ ] The account received a $12715.37 ACH payment flagged by blend  (`history:17854_811AF2F80`)

## `49365_812A09D40`  truth: SCATTER-GATHER; BIPARTITE; CYCLE; RANDOM; STACK

Predicted **CYCLE**, action **enhanced_review**, verified True, drafts 1.

> The account repeatedly exchanged ACH payments with the same counterparty, and the network contains a two-step cycle. The riskiest payment was received from that counterparty and was flagged at the blend score of 5.06. Although multiple models flagged payments and the network shows a cycle, the blend rank is 71, so enhanced review is appropriate rather than immediate escalation.

- [ ] The account's riskiest payment was a received ACH payment of 30749.49 from 119_812A09CF0 with a blend score of 5.06.  (`history:49365_812A09D40`)
- [ ] The network contains 1 two-step cycle.  (`subgraph:49365_812A09D40`)
- [ ] There is 1 counterparty with which the account has payments in both directions.  (`subgraph:49365_812A09D40`)
- [ ] The blend, xgb, and gnn flagged 6, 4, and 6 payments, respectively.  (`alert:49365_812A09D40`)
- [ ] The blend rank is 71.  (`alert:49365_812A09D40`)
- [ ] The receiver of the riskiest payment had already paid the sender before.  (`shap:49365_812A09D40`)

## `22164_8014621B0`  truth: FAN-IN

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 2.

> The account ranked 25 and had a risk score of 5.29, with each of the three models flagging 1 payment. Its riskiest payment was a flagged outgoing ACH transfer of 13916.8 Euro, and the neighbourhood contained 3 flagged payments. The available network evidence does not establish a clear fan-in, fan-out, cycle, or layered-chain pattern, so the account requires enhanced review rather than immediate escalation.

- [ ] The blend model ranked the account 25 with a risk score of 5.29.  (`alert:22164_8014621B0`)
- [ ] Each of the blend, XGB, and GNN models flagged 1 payment.  (`alert:22164_8014621B0`)
- [ ] The account made 5 outgoing payments and received 0 payments.  (`history:22164_8014621B0`)
- [ ] The account sent a flagged ACH payment of 13916.8 Euro to 11056_800E3D880.  (`history:22164_8014621B0`)
- [ ] The account had 3 direct payments sent, 2 direct payments received, and no detected two-step or three-step cycles.  (`subgraph:22164_8014621B0`)

## `11056_800E3D880`  truth: FAN-IN

Predicted **FAN-IN**, action **escalate_SAR**, verified True, drafts 1.

> The account receives payments from 13 different senders while sending to 6 different accounts, indicating a fan-in concentration. All three models flag payments, the account is ranked 26, and the network supports a nameable pattern, so the case warrants SAR escalation.

- [ ] 13 different accounts pay the account.  (`subgraph:11056_800E3D880`)
- [ ] The account sends payments to 6 different accounts.  (`subgraph:11056_800E3D880`)
- [ ] 8 flagged payments touch the account.  (`subgraph:11056_800E3D880`)
- [ ] The account is ranked 26.  (`alert:11056_800E3D880`)
- [ ] The riskiest payment was made by ACH transfer and had an amount of 13916.8.  (`shap:11056_800E3D880`)

## `13858_801820540`  truth: FAN-IN

Predicted **FAN-OUT**, action **escalate_SAR**, verified True, drafts 1.

> The account shows a fan-out pattern, with 7 direct payments sent to 3 different receivers and 0 different senders. The blend rank is 27, the risk score is 5.28, and each model flags 1 payment. The riskiest payment was a sent ACH payment of 9511.58 US Dollars to 14325_801A21E40.

- [ ] The account has 7 direct payments sent to 3 different receivers.  (`subgraph:13858_801820540`)
- [ ] The account has 0 different senders.  (`subgraph:13858_801820540`)
- [ ] There are 0 two-step cycles and 0 three-step cycles.  (`subgraph:13858_801820540`)
- [ ] Each model flags 1 payment.  (`alert:13858_801820540`)
- [ ] The account is ranked 27 with a risk score of 5.28.  (`alert:13858_801820540`)
- [ ] A sent payment of 9511.58 US Dollars by ACH to 14325_801A21E40 was flagged.  (`history:13858_801820540`)

## `48309_811C599A0`  truth: GATHER-SCATTER; FAN-OUT

Predicted **CYCLE**, action **escalate_SAR**, verified True, drafts 1.

> The account is ranked 19, and all three models identify flagged payments. The neighborhood shows reciprocal activity, with 3 two-step cycles and 2 three-step cycles. This supports escalation for a potential cycle pattern.

- [ ] The account is ranked 19 in the alert queue.  (`alert:48309_811C599A0`)
- [ ] The blend model flags 6 payments, the xgb model flags 4 payments, and the gnn model flags 5 payments.  (`alert:48309_811C599A0`)
- [ ] The network contains 3 two-step cycles and 2 three-step cycles.  (`subgraph:48309_811C599A0`)
- [ ] There are 3 counterparties with payments in both directions.  (`subgraph:48309_811C599A0`)

## `150240_812D22980`  truth: FAN-OUT

Predicted **CYCLE**, action **escalate_SAR**, verified True, drafts 1.

> The network shows a recurring loop pattern, with 3 two-step cycles and 2 three-step cycles involving the account. The blend and XGB assessments each identify 3 flagged payments, and the account is ranked 20. The riskiest payment was an ACH transfer to a receiver that had previously paid the sender.

- [ ] The account is ranked 20 in the alert queue and has a risk score of 5.32.  (`alert:150240_812D22980`)
- [ ] The blend and XGB assessments each identify 3 flagged payments, while the GNN assessment identifies 1 flagged payment.  (`alert:150240_812D22980`)
- [ ] The payment network shows 3 two-step cycles and 2 three-step cycles.  (`subgraph:150240_812D22980`)
- [ ] The account's riskiest payment was made by ACH transfer, and the receiver had already paid the sender before.  (`shap:150240_812D22980`)
- [ ] The riskiest payment had an amount of 74115.98 and was in the account's riskiest payment history.  (`shap:150240_812D22980`)

## `21611_808D6F2D0`  truth: FAN-OUT

Predicted **FAN-OUT**, action **escalate_SAR**, verified True, drafts 1.

> The network shows a fan-out pattern: the account sends to 15 different receivers while receiving from 3 different senders, with 1 two-step cycle present. Every model flagged multiple payments, and the blend rank is 46. The account therefore warrants escalation for further review under the named network pattern.

- [ ] The account sends payments to 15 different receivers.  (`subgraph:21611_808D6F2D0`)
- [ ] The account receives payments from 3 different senders.  (`subgraph:21611_808D6F2D0`)
- [ ] The network contains 1 two-step cycle.  (`subgraph:21611_808D6F2D0`)
- [ ] The blend model flagged 8 payments, the XGB model flagged 8 payments, and the GNN model flagged 7 payments.  (`alert:21611_808D6F2D0`)
- [ ] The blend rank is 46.  (`alert:21611_808D6F2D0`)
- [ ] The riskiest payment was sent by ACH transfer for 15190.56 in Swiss Franc and was flagged by the blend model.  (`history:21611_808D6F2D0`)

## `222965_80EFDDBB0`  truth: GATHER-SCATTER

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> The account ranked 3, and the blend, XGB, and GNN systems each flagged 1 payment. Its network has 1 distinct sender and 1 distinct receiver with 0 two-step cycles and 0 three-step cycles, so the available evidence does not establish a named transactional pattern. The riskiest payment was an ACH transfer of 16418.79 Euro, and the account should receive enhanced review rather than SAR escalation on the current evidence.

- [ ] The account has a risk score of 5.58 and a rank of 3.  (`alert:222965_80EFDDBB0`)
- [ ] The riskiest payment was an ACH transfer of 16418.79 Euro sent to 11_800924840 and was flagged by the blend system.  (`history:222965_80EFDDBB0`)
- [ ] The network shows 1 distinct sender and 1 distinct receiver.  (`subgraph:222965_80EFDDBB0`)
- [ ] There are 0 two-step cycles and 0 three-step cycles.  (`subgraph:222965_80EFDDBB0`)
- [ ] The blend, XGB, and GNN systems each flagged 1 payment.  (`alert:222965_80EFDDBB0`)
- [ ] Paid by ACH transfer was the strongest listed factor raising the risk of the riskiest payment, with a push of 2.7.  (`shap:222965_80EFDDBB0`)

## `11_800924840`  truth: GATHER-SCATTER

Predicted **GATHER-SCATTER**, action **escalate_SAR**, verified True, drafts 1.

> The account shows a gather-scatter pattern, with 15 different senders paying it and 10 different receivers receiving its payments. All three models flagged payments, and the account ranked 4 in the alert queue. A received ACH transfer of 16418.79 had a blend score of 5.58, matching the account's reported risk score.

- [ ] All three models flagged payments for the account.  (`alert:11_800924840`)
- [ ] 15 different senders paid the account.  (`subgraph:11_800924840`)
- [ ] The account paid 10 different receivers.  (`subgraph:11_800924840`)
- [ ] No payment cycles were identified.  (`subgraph:11_800924840`)
- [ ] A received ACH transfer of 16418.79 was flagged with a blend score of 5.58.  (`history:11_800924840`)

## `9371_8043A0FB0`  truth: GATHER-SCATTER

Predicted **FAN-OUT**, action **escalate_SAR**, verified True, drafts 1.

> The network shows a fan-out pattern, with the account paying many distinct receivers while also having multiple senders. The same flagged payment is identified by the blend, xgb, and gnn, and the account's blend rank is 5. The payment drivers point to ACH format, recent activity on both sides, and a payment amount of 9359.22.

- [ ] The account paid 17 different receivers.  (`subgraph:9371_8043A0FB0`)
- [ ] The account received payments from 11 different senders.  (`subgraph:9371_8043A0FB0`)
- [ ] The blend rank was 5, and the blend, xgb, and gnn each flagged 1 payment.  (`alert:9371_8043A0FB0`)
- [ ] The account's riskiest payment was by ACH transfer and had an amount of 9359.22.  (`shap:9371_8043A0FB0`)
- [ ] There were 0 two-step cycles and 0 three-step cycles.  (`subgraph:9371_8043A0FB0`)

## `15010_80362C640`  truth: RANDOM

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> The account's riskiest payment was an incoming ACH transfer of 13526.09 Euro and was flagged at 5.03. The surrounding network does not show a clear fan-in, fan-out, or cycle pattern, so the money-movement typology is unclear. The blend rank is 82 and the evidence does not identify a named network pattern, so enhanced review is recommended.

- [ ] The account received an ACH payment of 13526.09 Euro from 9665_804234190.  (`history:15010_80362C640`)
- [ ] The received payment of 13526.09 Euro was flagged and had a blend score of 5.03.  (`history:15010_80362C640`)
- [ ] The account received payments from 2 different accounts.  (`subgraph:15010_80362C640`)
- [ ] The account made payments to 2 different accounts.  (`subgraph:15010_80362C640`)
- [ ] The account's riskiest payment was paid by ACH transfer, which raised risk.  (`shap:15010_80362C640`)

## `9665_804234190`  truth: RANDOM

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> The network evidence does not identify a clear typology, as the account has inbound activity but no observed cycles or bidirectional counterparties. Its riskiest payment was an outgoing ACH payment that all three models flagged. Because the account's rank is 83, the case does not meet the stated escalation threshold and should receive enhanced review.

- [ ] The account's blend rank is 83 and its risk score is 5.03.  (`alert:9665_804234190`)
- [ ] Each of the three models flagged 1 payment.  (`alert:9665_804234190`)
- [ ] 4 different accounts paid the account, while it sent payments to 1 receiver.  (`subgraph:9665_804234190`)
- [ ] The account's riskiest payment was an outgoing ACH payment of 13526.09 Euro.  (`history:9665_804234190`)
- [ ] The most recent payment was a received Wire payment of 3672.2 Yen, and it was not flagged.  (`history:9665_804234190`)

## `23691_8021353D0`  truth: CYCLE; SCATTER-GATHER; GATHER-SCATTER; BIPARTITE; FAN-IN; RANDOM

Predicted **CYCLE**, action **enhanced_review**, verified True, drafts 1.

> The activity repeatedly moves between the account and the same counterparty, and the network shows 1 two-step cycle. The riskiest payment was an incoming ACH transfer of 14840.19 Euro, with prior reciprocal payment activity contributing to its risk profile. Although three assessment methods flagged multiple payments, the blend ranking of 99 requires enhanced review rather than immediate escalation.

- [ ] The riskiest payment was a received ACH transfer of 14840.19 Euro from 15231_80266F880, and it was flagged with a blend score of 4.97.  (`history:23691_8021353D0`)
- [ ] The account sent 5 payments and received 15 payments, of which 8 were flagged.  (`history:23691_8021353D0`)
- [ ] The payment network shows 1 two-step cycle and 0 three-step cycles.  (`subgraph:23691_8021353D0`)
- [ ] The account has 1 counterparty with payments in both directions.  (`subgraph:23691_8021353D0`)
- [ ] For the riskiest payment, the receiver had already paid the sender before.  (`shap:23691_8021353D0`)
- [ ] The blend ranking is 99, and three assessment methods flagged multiple payments, with counts of 6, 6, and 5.  (`alert:23691_8021353D0`)

## `222697_8087F4940`  truth: SCATTER-GATHER

Predicted **FAN-IN**, action **escalate_SAR**, verified True, drafts 1.

> The network shows a clear fan-in pattern: multiple different senders pay the account, while it has 1 distinct receiver. The account's riskiest payment was an ACH transfer of 17049.76. Each of the blend, xgb, and gnn flagged 1 payment, and the account holds rank 1.

- [ ] 5 different accounts pay the account, while it pays 1 different account.  (`subgraph:222697_8087F4940`)
- [ ] The account sent an ACH payment of 17049.76 to 15085_81121FAB0, and the blend flagged it.  (`history:222697_8087F4940`)
- [ ] The blend, xgb, and gnn each flagged 1 payment.  (`alert:222697_8087F4940`)
- [ ] The account is ranked 1.  (`alert:222697_8087F4940`)

## `15085_81121FAB0`  truth: SCATTER-GATHER

Predicted **FAN-IN**, action **escalate_SAR**, verified True, drafts 1.

> The account shows a clear fan-in pattern, with many different senders paying it while it has few outgoing receivers. Every scoring component flagged payments, and the account has a high blend ranking. The pattern supports escalation for SAR review.

- [ ] 15 different accounts paid the account, while it had only 1 distinct receiver.  (`subgraph:15085_81121FAB0`)
- [ ] The account received 36 direct payments and sent 12 direct payments.  (`subgraph:15085_81121FAB0`)
- [ ] The network showed 0 two-step cycles and 0 three-step cycles.  (`subgraph:15085_81121FAB0`)
- [ ] All three scoring components flagged payments, with 6, 6, and 5 flagged payments.  (`alert:15085_81121FAB0`)
- [ ] The riskiest payment was received by ACH transfer and had an amount of 17049.76.  (`shap:15085_81121FAB0`)

## `220_800118660`  truth: SCATTER-GATHER

Predicted **CYCLE**, action **escalate_SAR**, verified True, drafts 1.

> The account has a risk score of 5.54 and is ranked 7. The blend, xgb, and gnn models each flag a payment, and the network contains a 1 two-step cycle, supporting escalation for review.

- [ ] The account has a risk score of 5.54, is ranked 7, and the blend, xgb, and gnn models each flag 1 payment.  (`alert:220_800118660`)
- [ ] The network shows 1 two-step cycle and 0 three-step cycles.  (`subgraph:220_800118660`)
- [ ] The account receives payments from 9 distinct senders and sends payments to 4 distinct receivers.  (`subgraph:220_800118660`)
- [ ] The riskiest payment was paid by ACH transfer and had an amount of 17840.32.  (`shap:220_800118660`)
- [ ] The most recent payment was received in US Dollar by ACH for 17840.31 and was flagged.  (`history:220_800118660`)

## `11_8006345A0`  truth: STACK

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> All three models flagged 1 payment, and the blend ranking places the account at 17. However, the network has 1 distinct sender, 1 distinct receiver, and no identified loops or layered structure. The evidence therefore does not support a single named typology and warrants enhanced review.

- [ ] The account's riskiest payment has a blend risk score of 5.33.  (`alert:11_8006345A0`)
- [ ] Each of the three models flagged 1 payment.  (`alert:11_8006345A0`)
- [ ] The account's network has 1 distinct receiver.  (`subgraph:11_8006345A0`)
- [ ] The account's network has 1 distinct sender.  (`subgraph:11_8006345A0`)
- [ ] The network shows 0 two-step cycles and 0 three-step cycles.  (`subgraph:11_8006345A0`)
- [ ] The account's riskiest payment was sent by ACH for 19496.68 US Dollar.  (`history:11_8006345A0`)

## `11974_800FC2A60`  truth: STACK

Predicted **FAN-OUT**, action **escalate_SAR**, verified True, drafts 1.

> The account shows a fan-out pattern, paying 7 distinct receivers compared with 3 distinct senders. There are 0 two-step cycles and 0 three-step cycles, and the riskiest payment was a received US Dollar 19496.68 ACH payment. All three systems flagged 1 payment, and the account ranks 18, supporting escalation.

- [ ] The account's risk score is 5.33 and its rank is 18.  (`alert:11974_800FC2A60`)
- [ ] All three systems flagged 1 payment.  (`alert:11974_800FC2A60`)
- [ ] The network shows 7 distinct receivers and 3 distinct senders.  (`subgraph:11974_800FC2A60`)
- [ ] The account sent 82 direct payments and received 18 direct payments.  (`subgraph:11974_800FC2A60`)
- [ ] The account received a flagged US Dollar 19496.68 ACH payment.  (`history:11974_800FC2A60`)
- [ ] The riskiest payment was an ACH transfer, which raises risk.  (`shap:11974_800FC2A60`)

## `24515_80254C330`  truth: STACK

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> The account has 3 distinct senders and 3 distinct receivers, with 0 two-step cycles and 0 three-step cycles, so the observed network does not establish a specific laundering pattern. The history contains 17 received payments and 3 sent payments, including a flagged sent ACH payment of 9921.64 in Euro. Each risk model flagged 1 payment and the blend rank is 21, but the lack of a named network pattern supports enhanced review rather than immediate escalation.

- [ ] Each risk model flagged 1 payment.  (`alert:24515_80254C330`)
- [ ] The blend rank was 21.  (`alert:24515_80254C330`)
- [ ] The account had 3 distinct senders and 3 distinct receivers, with 0 two-step cycles and 0 three-step cycles.  (`subgraph:24515_80254C330`)
- [ ] A sent ACH payment of 9921.64 in Euro was flagged with a blend score of 5.31.  (`history:24515_80254C330`)
- [ ] The payment history contained 17 received payments and 3 sent payments.  (`history:24515_80254C330`)

## `312002_805B959D0`  truth: clean

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> The account's rank is 174, and multiple assessments flag a payment involving 4324.31 US Dollars. The network does not show a clear fan-in, fan-out concentration, cycle, or layered payment structure. The available evidence does not meet the escalation criteria, so enhanced review is appropriate.

- [ ] The account's risk score is 4.56.  (`alert:312002_805B959D0`)
- [ ] The blend and XGBoost assessments each flag a payment, while the graph-based assessment does not.  (`alert:312002_805B959D0`)
- [ ] The account's rank is 174.  (`alert:312002_805B959D0`)
- [ ] The network shows payments to 2 different receivers and no payments from different senders.  (`subgraph:312002_805B959D0`)
- [ ] No two-step or three-step cycles were found in the network.  (`subgraph:312002_805B959D0`)
- [ ] The riskiest payment was sent to 1124_805BA8930 by ACH for 4324.31 US Dollars and was flagged by the blend assessment.  (`history:312002_805B959D0`)

## `16_8058AF4B0`  truth: clean

Predicted **FAN-IN**, action **enhanced_review**, verified True, drafts 1.

> The account received 12 direct payments from 9 different accounts while making 2 direct payments, indicating a fan-in pattern. The blend and XGBoost checks each flagged 1 payment, and the riskiest payment used ACH. The blend rank was 177, so the case warrants enhanced review rather than immediate escalation.

- [ ] The account was paid by 9 different accounts.  (`subgraph:16_8058AF4B0`)
- [ ] The account made 2 direct payments and received 12 direct payments.  (`subgraph:16_8058AF4B0`)
- [ ] The blend and XGBoost checks each flagged 1 payment, while the graph check flagged 0 payments.  (`alert:16_8058AF4B0`)
- [ ] The riskiest payment was paid by ACH transfer, had an amount of 41043136.31, and the sender's time since the account's previous activity was 0.0 hours.  (`shap:16_8058AF4B0`)
- [ ] The most recent payment was received from 10_80012FD40 in Rupee by ACH for 41043106.82 and was flagged by the blend check.  (`history:16_8058AF4B0`)

## `10_80012FD40`  truth: clean

Predicted **FAN-IN**, action **enhanced_review**, verified True, drafts 1.

> The account is ranked 178 and received payments from 8 different accounts while paying 4 different accounts, supporting a fan-in pattern. No two-step or three-step cycles were found, and the blend and xgb analyses each flagged 1 payment. The riskiest payment was an outgoing ACH transfer of 41043106.82 Rupee, so the account warrants enhanced review rather than immediate escalation.

- [ ] The account received payments from 8 different accounts.  (`subgraph:10_80012FD40`)
- [ ] The account paid 4 different accounts.  (`subgraph:10_80012FD40`)
- [ ] The network contained 0 two-step cycles and 0 three-step cycles.  (`subgraph:10_80012FD40`)
- [ ] The blend and xgb analyses each flagged 1 payment, while the gnn analysis flagged 0 payments.  (`alert:10_80012FD40`)
- [ ] The account's riskiest payment was an outgoing ACH transfer of 41043106.82 Rupee that was flagged by blend.  (`history:10_80012FD40`)

## `331557_810B45D20`  truth: clean

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> The account sent 6 payments and received 0 payments, with 1 payment flagged. The riskiest payment was a sent ACH payment of 654.67 US Dollar that was flagged. The available network shows 1 distinct receiver, 0 distinct senders, and no 2-step or 3-step cycles, so it does not establish a named laundering pattern.

- [ ] The account sent 6 payments and received 0 payments.  (`history:331557_810B45D20`)
- [ ] The riskiest payment was a sent ACH payment of 654.67 US Dollar to account 2845_810B45BE0 and was flagged.  (`history:331557_810B45D20`)
- [ ] The network has 1 distinct receiver and 0 distinct senders.  (`subgraph:331557_810B45D20`)
- [ ] The network has 0 two-step cycles and 0 three-step cycles.  (`subgraph:331557_810B45D20`)
- [ ] The sender had made 2.0 payments in the last 7 days.  (`shap:331557_810B45D20`)

## `2845_810B45BE0`  truth: clean

Predicted **FAN-IN**, action **enhanced_review**, verified True, drafts 2.

> The account received payments from 7 different accounts and paid 2 different accounts, indicating a fan-in pattern. Its riskiest payment was a received ACH transfer of 654.67 US Dollar, and the ACH format and sender activity were identified as increasing risk. The account ranked 180, so enhanced review is appropriate.

- [ ] 7 different accounts pay the account, while it pays 2 different accounts.  (`subgraph:2845_810B45BE0`)
- [ ] The account's riskiest payment was a received ACH transfer of 654.67 US Dollar that was flagged.  (`history:2845_810B45BE0`)
- [ ] The ACH transfer increased risk by 2.45, and the sender's time since the account's previous activity increased risk by 2.14.  (`shap:2845_810B45BE0`)
- [ ] The blend and XGB assessments each flagged 1 payment, while the GNN assessment flagged 0 payments.  (`alert:2845_810B45BE0`)

## `11974_8057E3520`  truth: clean

Predicted **UNCLEAR**, action **enhanced_review**, verified True, drafts 1.

> The account's riskiest payment was an ACH transfer of 3265.6 US Dollar to 3149_8057E3700, and it was flagged. Activity is concentrated around 1 distinct sender and 1 distinct receiver, with no detected two-step or three-step cycles. The blend and XGBoost models flag 1 payment each, but the evidence does not identify a specific network pattern.

- [ ] The account has a risk score of 4.46 and ranks 181 among 200 accounts in the queue.  (`alert:11974_8057E3520`)
- [ ] The blend and XGBoost models each flag 1 payment, while the GNN model flags 0 payments.  (`alert:11974_8057E3520`)
- [ ] The network shows 1 distinct receiver and 1 distinct sender, with no two-step or three-step cycles detected.  (`subgraph:11974_8057E3520`)
- [ ] The account's riskiest payment was 3265.6 US Dollar sent by ACH, and it was flagged with a blend score of 4.46.  (`history:11974_8057E3520`)
