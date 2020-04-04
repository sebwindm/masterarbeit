from src import class_Order
import random


def sort():
    list = []
    for i in range(10):
        count_of_generated_orders = random.randrange(1,1000)
        list.append(class_Order.Order(count_of_generated_orders, 0, 0))
    print("Vorher:")
    for i in list:
        print(i.orderID)

    list.sort(key=lambda x: x.orderID, reverse=False)
    print("Nachher:")
    for i in list:
        print(i.orderID)
    return
sort()
