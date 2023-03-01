# only used in local, for easily debug code.
ps -ef|grep main.py|grep -v grep|awk '{print $2}'|xargs kill -9
