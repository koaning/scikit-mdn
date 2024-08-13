from skmdn import MixtureDensityEstimator

docs = ''

# Loop through each attribute in the MixtureDensityEstimator class
for attribute_name in dir(MixtureDensityEstimator):
    # Get the attribute
    attribute = getattr(MixtureDensityEstimator, attribute_name)
    # Check if the attribute is a callable method and not a built-in method
    if callable(attribute) and not attribute_name.startswith('_'):
        docs += f'#### {attribute_name}\n'
        docs += f'{attribute.__doc__}\n\n'
        
print(docs)