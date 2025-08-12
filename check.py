a=["h","e","l","l","o"]
b=0
c=len(a)-1
while b<c:
    a[b], a[c] = a[c], a[b]
    b+=1
    c-=1
print(a)
