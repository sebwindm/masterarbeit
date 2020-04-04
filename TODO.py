"""
TODO:
- braucht es sort_order_pool_by_due_date()? aktuell werden neue orders hinten an order pool angehängt
    darum ist die sortierung eh schon richtig

- Kostenberechnung: am Ende jeder Periode festlegen (z.B. 1€ für jede Order die in WIP_A war)
    oder genauer in jedem Step (1/960€ pro Step)?


- Machine processing times sind jetzt in steps angegeben

- Order arrivals nach exp/unif. distribution fehlt komplett --> implementiert wurde Gleichverteilung












Simulation setup von Stefan:

6 products: 1-6

6 machines: A-F

960 minutes = 1 period
run for 8000 periods and 30 (?) replications
first 1000 periods are warm-up

due date is always order arrival + 10

processing times distribution: exponential or uniform
                high variance:          low variance:
utilisation 70%     Exp(135)                U(95,175)
utilisation 80%     Exp(118)                U(78,158)

before releasing, orders are sorted by due date inside the order pool

wip cost: 1 € per order per period (measured at end of period)
fgi cost: 4 € "
backorder cost: 16 € "

            High variance   Low variance
Machine A   Exp(80)             U(30,130)
Machine B   Exp(160)            U(80,240)
Machine C   Exp(155)            U(50,260)
Machine D   Exp(210)            U(50,370)
Machine E   Exp(285)            U(200,370)
Machine F   Exp(215)            U(110,320)

"""



"""
Was implementiert wurde:
- Simulation mit steps, alle Aktionen wie order generation, processing etc können ein mal pro step stattfinden
- Eine Periode = 960 steps, zu Testzwecken läuft es gerade für 2 Perioden
- Maschine mit processing time von 30 benötigt also 30 steps um eine Order fertigzustellen,
    andere Aktionen wie Bewegen einer Order von WIP zu Maschine geht sofort (aber nur 1x pro Step)
- Order generation alle X steps, wobei X eine Zufallszahl (fester Intervall) ist die nach jeder neuen
    Order-Erstellung neu ausgewürfelt wird. DueDate = arrival time + 10 periods
- Order release aller Orders mit jeder neuen Periode, also alle 960 steps werden alle orders releast. 
    Order pool ist aber bereits nach DueDate sortiert
- Processing times der Maschinen Zufallszahl aus Intervall, aber wird nur 1x am Anfang ausgewürfelt
- Kosten von WIP/FGI/lateness werden mit jedem step, den eine Order wartet, um 1/4/16 erhöht
- Routing komplett eingebaut
"""
