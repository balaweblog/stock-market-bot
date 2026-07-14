import os
os.environ['DRY_RUN'] = 'true'
import main
main.main('eod', use_llm=False)
size = os.path.getsize('report.html')
print(f'report.html size: {size} bytes ({size/1024:.1f} KB)')
with open('report.html') as f:
    content = f.read()
print(f'Commodities in HTML: {"Commodities" in content}')
print(f'Gold card in HTML: {"Gold (22K)" in content}')
print(f'Silver card in HTML: {"XAG/INR" in content}')
# Gmail clips at ~102KB
if size > 102400:
    print(f'*** WARNING: Email is {size/1024:.1f} KB — Gmail will CLIP at 102 KB! ***')
    print('*** Commodities section at the END will be cut off. ***')
else:
    print(f'Email size is OK for Gmail (under 102 KB)')
