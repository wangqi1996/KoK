a1=H-679-IV-de
a2=H-717-Annex-de
a3=lamictal_bi_de
a4=menitorix_bi_de
a5=Doxyprex_Background_Information-de
a6=norfloxacin-bi-de
a7=uman_big_q_a_de
a8=sabumalin_q_a_de
a9=etoricoxib-arcoxia-bi-de
a10=sanohex_q_a_de
a11=implanon_Q_A_de
a12=emea-2006-0258-00-00-ende
a13=V-121-de1
a14=H-902-WQ_A-de
a15=V-141-de1
a16=V-137-de1
a17=Hexavac-H-298-Z-28-de
a18=H-891-de1
a19=V-030-de1
a20=V-107-de1
a21=compagel-v-a-33-030-de
a22=V-126-de1

for f in $a1 $a2 $a3 $a4 $a5; do
  bash scripts/medical-process/bpe_and_bin.sh $f 30 50
done

for f in $a6 $a7 $a8 $a9 $a10 $a11 $a12 $a13 $a14 $a15 $a16 $a17 $a18 $a19 $a20 $a21 $a22; do
  bash scripts/medical-process/bpe_and_bin.sh $f 80 50
done

b1=093604de1
b2=H-741-de1
b3=H-897-de1
b4=H-933-de1
b5=tritazide_q_a_de
b6=V-133-de1
b7=H-915-de1
b8=H-725-de1
b9=400803de1
b10=H-890-de1
b11=Veralipride-H-A-31-788-de
b12=Belanette-AnnexI-III-de
b13=V-A-35-029-de
b14=112901de4

#for f in $b1 $b2 $b3 $b4 $b5 $b6 $b7 $b8 $b9 $b10 $b11; do
#  bash scripts/medical-process/bpe_and_bin.sh $f 80 100
#done
#
#for f in $b12 $b13 $b14; do
#  bash scripts/medical-process/bpe_and_bin.sh $f 130 100
#done

c1=49533907de
c2=implanon_annexI_IV_de
c3=EMEA-CVMP-82633-2007-de
c4=sanohex_annexI_III_de
c5=V-048-PI-de
c6=V-041-PI-de
c7=V-047-PI-de
#bash scripts/medical-process/bpe_and_bin.sh $c1 130 200
#bash scripts/medical-process/bpe_and_bin.sh $c2 130 200
#bash scripts/medical-process/bpe_and_bin.sh $c3 180 200
#bash scripts/medical-process/bpe_and_bin.sh $c4 180 200
#bash scripts/medical-process/bpe_and_bin.sh $c5 230 200
#bash scripts/medical-process/bpe_and_bin.sh $c6 230 200
#bash scripts/medical-process/bpe_and_bin.sh $c7 230 200

d1=V-105-PI-de
d2=H-391-PI-de
d3=H-668-PI-de
d4=H-960-PI-de
#bash scripts/medical-process/bpe_and_bin.sh $d1 280 500
#bash scripts/medical-process/bpe_and_bin.sh $d2 380 500
#bash scripts/medical-process/bpe_and_bin.sh $d3 480 500
#bash scripts/medical-process/bpe_and_bin.sh $d4 530 500

e1=H-884-PI-de
e2=H-287-PI-de
e3=H-890-PI-de
e4=H-273-PI-de
e5=H-115-PI-de

#bash scripts/medical-process/bpe_and_bin.sh $e1 580 1000
#bash scripts/medical-process/bpe_and_bin.sh $e2 680 1000
#bash scripts/medical-process/bpe_and_bin.sh $e3 780 1000
#bash scripts/medical-process/bpe_and_bin.sh $e4 880 1000
#bash scripts/medical-process/bpe_and_bin.sh $e5 980 1000
