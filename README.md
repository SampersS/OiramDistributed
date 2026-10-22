# OiramDistributed

## 2 delen
- Reinforcement execution, and modification.
- distribution of execution models, harvesting results.

## network model
We hebben 1 hoofd node, deze beheert de taken en verbindingen van de slave nodes.<br>
Om de bepaalde tijd komt er een request uit de hoofd node om te checken op de vooruitgang van de slaves, deze kunnen dan hun vooruitgang delen. Dan zal de hoofd node beslissen om een nieuwe batch te maken of niet. implementatie in ./reiforcementProtocol.md.

## discovery
Via dhcp weet de hoofd node welke apparaten er op het netwerk zitten, dan komt er een request voor niewue apparaten om na te gaan of ze reinforcement learning nodes zijn.

## requests
niet via broadcast, er wordt een lijst met verbonden apparaten bijgehouden.