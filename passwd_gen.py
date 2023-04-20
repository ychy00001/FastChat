import random

invite_code_list_file = "./passwd"
with open(invite_code_list_file, "w") as file:
    for i in range(0, 50):
        item = "".join(random.sample('1234567890', 6)) + "\n"
        file.write(item)
    file.write("jiangziya\n")
    file.write("qwer")
