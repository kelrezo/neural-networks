import sys
a=sys.argv[1]
#b=sys.argv[2]
#c=sys.argv[3]
f1='=IF(AND(EXACT(A2:'+a+'2,source!A1:'+a+'1)),"pass","fail")'
#f2 = '=COUNTIF('+b+'2:'+b+'101,"pass")/100'
#f3 = '=1-'+c +'2'
print(f1)
#print(f2)
#print(f3)

