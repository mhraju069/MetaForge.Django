from django.contrib import admin
from .models import *
from unfold.admin import ModelAdmin
# Register your models here.

admin.site.register(User, ModelAdmin)
admin.site.register(Company, ModelAdmin)
