# import requests, bs4

# i = 1
# authors = set()
# while True:
#     try:
#             result = requests.get(("http://quotes.toscrape.com/page/{}/").format(i))

#             soup = bs4.BeautifulSoup(result.text, "lxml")

#             for k in soup.select(".author"):
#                 authors.add(k.text)
                
#             for j in authors:
#                 print(j)
                
#             i = i + 1
        
#     except:
#         print(result.text)
#         print("asdasf")
#         break
    
from PIL import Image

img_word_matrix = Image.open("Python/word_matrix.png")
img_mask        = Image.open("Python/mask.png")

img_mask =  img_mask.resize((1015,559))

img_word_matrix.putalpha(50)

img_mask.paste(im = img_word_matrix, box = (0,0),mask = img_word_matrix)

img_mask.show()

