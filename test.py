from pipeline.capabilities.classification import get_facet, list_facets

# See what's available
list_facets()  # ['alerts', 'recommendations']

# Get a facet and use it
# alerts = get_facet("alerts")
# prompt = alerts.prompt_fn("test")
# print(prompt)
