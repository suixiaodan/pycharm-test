#!/bin/sh

#------------------------------------------------------------------------------
# 函数: CheckProcess
# 功能: 检查一个进程是否存在
# 参数: $1 --- 要检查的进程名称
# 返回: 如果存在返回1, 否则返回0.
#------------------------------------------------------------------------------
CheckProcess()
{
# 检查输入的参数是否有效
#  if [ "$1" = "" ];
#  then
#    return 1
#  fi

  #$PROCESS_NUM获取指定进程名的数目，等于0返回1，表示没有该进程运行，不为0返回0，表示该进程在运行，等待等待
  PROCESS_NUM=`ps -ef | grep "$1" | grep -v "grep" | wc -l`
  if [ $PROCESS_NUM -eq 0 ];
  then
    return 1
  else
    return 0
  fi
}

# 检查test实例是否已经存在
while [ 1 ] ; do
 if CheckProcess "31797" | CheckProcess "31797" | CheckProcess "31797"
 then
   return 1
 fi

 if CheckProcess "run_start.sh"
 then
# 杀死所有test进程，可换任意你需要执行的操作
# killall -9 test
   exec ./AutoRun.sh &
 else
   return 0
 fi
 sleep 1
done