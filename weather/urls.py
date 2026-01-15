from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate_forecast/', views.generate_forecast, name='generate_forecast'),
    path('get_alerts/', views.get_alerts, name='get_alerts'),
]