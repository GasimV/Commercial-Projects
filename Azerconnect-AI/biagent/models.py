from django.db import models

class Customer(models.Model):
    name = models.CharField(max_length=100)
    city = models.CharField(max_length=100)

    class Meta:
        db_table = "customers"

class Order(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    order_date = models.DateField()
    payment_type = models.CharField(max_length=50)
    amount = models.FloatField()

    class Meta:
        db_table = "orders"  # Avoid conflict with SQL reserved word

class Review(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    score = models.FloatField()

    class Meta:
        db_table = "reviews"

