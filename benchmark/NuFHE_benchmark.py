import random
import nufhe
import time

# thr = any_api().Thread.create(interactive=True)

ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()

vm = ctx.make_virtual_machine(cloud_key)


def addBits(r, a, b, carry):
    # Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, carry)
    # Xor(t2[0], b, carry[0])
    t2 = vm.gate_xor(b, carry)

    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(a, t2)
    # And(t1[0], t1[0], t2[0])
    t1 = vm.gate_and(t1, t2)

    # Xor(r[1], carry[0], t1[0])
    r[1] = vm.gate_xor(carry, t1)

    return r


def addNumbers(ctA, ctB, nBits):
    ctRes = [[vm.empty_ciphertext((1,))] for i in range(nBits)]
    # carry = vm.empty_ciphertext((1,))
    bitResult = [[vm.empty_ciphertext((1,))] for i in range(2)]
    ctRes[0] = vm.gate_xor(ctA[0], ctB[0])
    # Xor(ctRes[0], ctA[0], ctB[0])
    carry = vm.gate_and(ctA[0], ctB[0])
    # And(carry[0], ctA[0], ctB[0])
    for i in range(1, nBits):
        bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
        # Copy(ctRes[i], bitResult[0]);
        ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])

        # Copy(carry[0], bitResult[1])
        carry = nufhe.LweSampleArray.copy(bitResult[1])

    return ctRes


def mulNumbers(ctA, ctB, secret_key, input_bits, output_bits):
    result = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    # [[vm.empty_ciphertext((1,))] for _ in range(output_bits)]
    # andRes = [[vm.empty_ciphertext((1,))] for _ in range(input_bits)]

    for i in range(input_bits):
        andResLeft = [ctx.encrypt(secret_key, [False]) for _ in
                      range(output_bits)]
        for j in range(input_bits):
            andResLeft[j + i] = vm.gate_and(ctA[j], ctB[i])
            # andResLeft[j + i] = nufhe.LweSampleArray.copy(andRes[j])
        result = addNumbers(andResLeft, result, output_bits)

    return result


if __name__ == '__main__':
    # sizes = [12,20,24,28]
    # for size in sizes:
    #     # size = 2 ** i
    #     # ***Gate wise ***
    #
    #     times = 5
    #     bits1 = [random.choice([False, True]) for i in range(size)]
    #     bits2 = [random.choice([False, True]) for i in range(size)]
    #     reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]
    #
    #     ciphertext1 = ctx.encrypt(secret_key, bits1)
    #     ciphertext2 = ctx.encrypt(secret_key, bits2)
    #     diff = 0
    #     for _ in range(times):
    #         start_time = time.time()
    #         result = vm.gate_and(ciphertext1, ciphertext2)
    #         diff += time.time() - start_time
    #     print(size, " ", diff / times)
        # result_bits = ctx.decrypt(secret_key, result)

    size=16
    bits = [[False] for i in range(size - 2)]
    zeros = [[False] for i in range(size)]

    bits1 = [[True]] + [[False]] + bits
    bits2 = [[True]] + [[True]] + bits
    ciphertext1 = [[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext2 = [[vm.empty_ciphertext((1,))] for i in range(size)]
    for i in range(size):
        ciphertext1[i] = ctx.encrypt(secret_key, bits1[i])
        ciphertext2[i] = ctx.encrypt(secret_key, bits2[i])
    start_time = time.time()
    # result = addNumbers(ciphertext1, ciphertext2, size)
    result = mulNumbers(ciphertext1, ciphertext2, secret_key, size, size * 2)
    print(time.time() - start_time)

    result_bits = [ctx.decrypt(secret_key, result[i]) for i in range(size * 2)]

    print(result_bits)
