from django.contrib import admin
from .models import *
from unfold.admin import ModelAdmin
# Register your models here.

admin.site.register(SocialAccount,ModelAdmin)
admin.site.register(SocialPost,ModelAdmin)
admin.site.register(PostMedia,ModelAdmin)

