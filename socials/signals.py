from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import SocialPost, PostMedia
from .helper import train_post_embedding, train_post_image_hash, detect_is_product

@receiver(post_save, sender=SocialPost)
def handle_post_training(sender, instance, created, **kwargs):
    """
    On new post save:
    1. Detect if it's a product post using AI.
    2. Only if it IS a product, generate semantic embedding vector.
    """
    if created or instance.is_product is None:
        # Step A: AI checks if this is a product post
        is_prod = detect_is_product(instance)
        # Save the result back to the post (avoid signal loop with update)
        SocialPost.objects.filter(id=instance.id).update(is_product=is_prod)

        # Step B: Only train vector if it's actually a product
        if is_prod and not instance.vector:
            # Reload to get fresh instance with the flag set
            fresh = SocialPost.objects.get(id=instance.id)
            train_post_embedding(fresh)


@receiver(post_save, sender=PostMedia)
def handle_media_image_hash(sender, instance, created, **kwargs):
    """
    Automatically generates pHash visual fingerprint for an image
    whenever a new PostMedia is saved.
    """
    if created and not instance.image_hash:
        train_post_image_hash(instance.post)
