from django.contrib import admin
from .models import SocialAccount
from unfold.admin import ModelAdmin
# Register your models here.

admin.site.register(SocialAccount,ModelAdmin)
