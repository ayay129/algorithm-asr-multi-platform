nohup bash -c '
while true; do
echo "===== $(date "+%F %T") =====" >> largev3.log
ps -eo pid,%cpu,%mem,vsz,rss,args --sort=-rss | grep large-v3 | grep -v grep | awk "{printf \"PID=%s CPU=%s%% MEM=%s%% VIRT=%.2fG RES=%.2fG CMD=%s\\n\", \$1,\$2,\$3,\$4/1024/1024,\$5/1024/1024,\$6}" >> largev3.log
sleep 30
done
' >/dev/null 2>&1 &