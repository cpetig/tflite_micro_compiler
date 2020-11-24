set arch riscv:rv32
target extended-remote localhost:3333
set remotetimeout 2000
set logging on
break hello_world2.cc:43
break hello_world2.cc:44
commands 2
print success
quit success
end
delete main
run
continue
