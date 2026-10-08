"""Оновлює OAuth access-token Kaggle через refresh-token (CLI сам цього не робить і падає з 'Authentication required')."""
from kagglesdk import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials

creds = KaggleCredentials.load(KaggleClient())
creds.refresh_access_token()
creds.save()
