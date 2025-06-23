FILE = gpt2.c
CC_FLAGS = -Wall -Wno-implicit-function-declaration -std=c99 -g
.PHONY: clean build
clean:		# Remove all generated files	
	rm -f out *.o *.dSYM
build: clean
	gcc -w -Wincompatible-pointer-types -g -o out $(FILE) -lm
run: build	
	./out 5 "I am the king of Jungle. My name is"
profile:
	valgrind --leak-check=full --track-origins=yes ./out