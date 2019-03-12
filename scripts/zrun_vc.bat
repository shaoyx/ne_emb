python -u -m vcsample > result/app001_email.txt ^
--input mydata/email_con_edge.txt ^
--output result/email_con_app001_emd.txt ^
--label-file mydata/email_con_label.txt ^
--graph-format edgelist ^
--model-v app ^
--model-c app ^
--app-jump-factor 0.3 ^
--window-size 10 ^
--exp-times 5 ^
--classification ^
--clf-ratio 0.5 ^
--epochs 5 ^
--epoch-fac 1000 ^
--combine 0.5
pause

