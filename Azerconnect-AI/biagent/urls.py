from django.urls import path
from . import views

urlpatterns = [
    path('query/', views.handle_query, name='handle_query'),
    path('', views.home, name='home'),  # Serve home.html at root
]

