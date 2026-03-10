from django.apps import AppConfig

class SocialsConfig(AppConfig):
    name = 'socials'

    def ready(self):
        # Register signals for automated AI training
        import socials.signals
