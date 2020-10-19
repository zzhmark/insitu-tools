import fire

def order_by_length(*items, f=False):
  """Orders items by length, breaking ties alphabetically."""
  sorted_items = sorted(items, key=lambda item: (len(str(item)), str(item))) if f else items
  return ' '.join(sorted_items)

if __name__ == '__main__':
  fire.Fire(order_by_length)