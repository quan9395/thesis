import sys

tokens = []
current_token = None
pos = -1
varMap = {}
final_result = 0

class Number:
    def __init__(self, value, type):
        self.type = type
        self.value = value

def error():
    print('Error')
    exit()

def advance():
    global pos, current_token
    pos += 1
    if pos >= len(tokens):
        current_token = 'EOF'
    else:
        current_token = tokens[pos]

def back():
    global pos, current_token
    pos -= 1
    current_token = tokens[pos]

def prog():
    global varMap
    while pos < len(tokens) - 2:
        let_in_end()
        varMap.clear()
    if current_token == 'EOF':
        exit(0)
    else:
        error()

def let_in_end():
    # print('letinend', current_token)
    if(current_token == 'let'):
        advance()
    else:
        error()
    while(current_token != 'in'):
        decl()
    advance()
    type = check_type(current_token)
    final_result = expr(type)
    if(current_token == 'end'):
        advance()
    else:
        error()
    if current_token == ';':
        advance()
        print(final_result.value)
        return
    else:
        error()

def check_type(type):
    if type == 'int' or type == 'real':
        advance()
        return type
    else:
        error()

def decl():
    # print('decl', current_token)
    global varMap
    varName, expValue, type = None, None, None
    varName = current_token
    advance()
    if(current_token == ':'):
        advance()
        type = check_type(current_token)
        # print(current_token)
        if current_token == '=':
            advance()
            expValue = expr(type)
            if(current_token == ';'):
                advance()
            else:
                error()
            
        else:
            error()
    else:
        error()
    varMap[varName] = Number(expValue.value, type)
    return

def expr(type):
    # print('expr',current_token)
    # if current_token == '(':
    #     advance()
    #     if current_token == 'if':
    #         advance()
    #         lhs = lookup(current_token, type)
    #         advance()
    #         if current_token == '<':
    #             advance()
    #             rhs = lookup(current_token, type)
    #             if (lhs.value < rhs.value):
    #                 advance()
                
    #     else:
    #         back()

    lhs = term(type)
    # advance()
    while True:
        if current_token == '+':
            advance()
            rhs = term(type)
            lhs = compute(lhs, '+', rhs)
        elif current_token == '-':
            advance()
            rhs = term(type)
            lhs = compute(lhs, '-', rhs)
        elif current_token == ')' or current_token == ';' or current_token == 'end':
            break
        else:
            error()
    # advance()
    return lhs

def term(type):
    # print('term', current_token)
    lhs = factor(type)
    while True:
        if current_token == '*':
            advance()
            rhs = factor(type)
            lhs = compute(lhs, '*', rhs)
        elif current_token == '/':
            advance()
            rhs = factor(type)
            lhs = compute(lhs, '/', rhs)
        else:
            break
    # advance()
    return lhs

def factor(type):
    # print('factor', current_token)
    result = []
    if current_token == '(':
        advance()
        val = expr(type)
        result = Number(val.value, type)
        # advance()
        if current_token == ')':
            advance()
        else:
            error()
    elif current_token == 'int' or current_token == 'real':
        type = check_type(current_token)
        if current_token == '(':
            advance()
            val = lookup(current_token, type)
            advance()
            result = Number(val.value, type)
            if current_token == ')':
                advance()
            else:
                error()
        else:
            error()
    else:
        result = lookup(current_token, type)
        advance()
    return result

def compute(lhs, op, rhs):
    if lhs.type == rhs.type:
        if op == '+':
            lhs.value += rhs.value
            return lhs
        elif op == '-':
            lhs.value -= rhs.value
            return lhs
        elif op == '*':
            lhs.value *= rhs.value
            return lhs
        elif op == '/':
            lhs.value /= rhs.value
            return lhs
        else:
            error()
    else:
        error()

def lookup(var, type):
    # print('lookup', current_token)
    if var in varMap:
        # advance()
        return varMap[var]
    else:
        if type == 'int':
            try:
                result = Number(int(var), 'int')
            # advance()
            except ValueError:
                error()
            return result
        elif type == 'real':
            try:
                result = Number(float(var), 'real')
            except ValueError:
                error()
            # advance()
            return result
    

# if __name__ == '__main__':
#     if (len(sys.argv)) != 2:
#         print("Wrong input command, Please use format: python3 parser_2729679.py input_file_name")
#     else:
#         try:
#             in_fp = open(sys.argv[1], 'r')
#         except:
#             print("Error opening file")
#             sys.exit(1)
#         tokens = in_fp.read().split()
#         print(tokens)
#         advance()
#         prog()

import sys

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Invalid command: please use format: python3 parser2729679.py inputfile")
        exit()
    in_fp = open(sys.argv[1], 'r')
    tokens = in_fp.read().split()
    advance()
    prog()
