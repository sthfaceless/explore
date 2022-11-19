import argparse
import datetime
import platform
import socket


def traceroute(dest, timeout=2, max_ttl=128, port=33434, num_tries=3):
    curr_ttl = 1
    send_package = b"Hello, world!"
    ip = socket.gethostbyname(dest)
    curr = '127.0.0.1'
    print(f"Start traceroute to {dest}, max length {max_ttl}, package size {len(send_package)} bytes "
          f"with timeout {timeout}s")

    while curr_ttl <= max_ttl:

        # initialize icmp receiver at localhost
        icmp_receiver = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        icmp_receiver.settimeout(timeout)

        # initialize package sender and set ttl
        udp_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        opt_key = 4 if 'windows' in platform.platform().lower() else 2
        udp_sender.setsockopt(socket.IPPROTO_HOPOPTS, opt_key, curr_ttl)

        for _ in range(num_tries):
            try:
                # measure send package time
                start = datetime.datetime.now()
                udp_sender.sendto(send_package, (ip, port))
                data, address = icmp_receiver.recvfrom(1024)
                response_ip = f'{data[12]}.{data[13]}.{data[14]}.{data[15]}'
                try:
                    curr = socket.gethostbyaddr(address[0])[0]
                except socket.error:
                    curr = address[0]
                print(f'{curr} ({response_ip}) {(datetime.datetime.now() - start).microseconds / 1000:.2f}ms', end='')
                break
            except socket.error as err:
                print('*', end=' ')
        print()
        icmp_receiver.close()
        udp_sender.close()

        # make path longer
        curr_ttl += 1

        # finish if we achieved destination
        if curr == dest or curr == ip:
            break


# to run the script just type 'python traceroute.py <host_name>'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traces route to host")
    parser.add_argument("host", default="ya.ru")
    parser.add_argument("--timeout", type=int, default=2)
    parser.add_argument("--maxttl", type=int, default=128)
    parser.add_argument("--tries", type=int, default=3)
    args = parser.parse_args()
    traceroute(args.host, timeout=args.timeout, max_ttl=args.maxttl, num_tries=args.tries)
