from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSSwitch, Controller
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import threading
import time
import http.server
import socketserver

class DHCPAttackTopo(Topo):
    def build(self):
        switch = self.addSwitch('s1')

        h1 = self.addHost('h1', ip='10.0.0.10/24')
        dhcp = self.addHost('dhcp', ip='10.0.0.50/24')
        evil = self.addHost('evil', ip='10.0.0.66/24')

        self.addLink(h1, switch)
        self.addLink(dhcp, switch, delay='500ms')
        self.addLink(evil, switch)

def start_web_server(port, content):
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

    with socketserver.TCPServer(('0.0.0.0', port), CustomHTTPRequestHandler) as httpd:
        httpd.serve_forever()

def dhcp_attack():
    topo = DHCPAttackTopo()
    net = Mininet(topo=topo, switch=OVSSwitch, controller=Controller, link=TCLink)
    net.start()

    h1 = net.get('h1')
    evil = net.get('evil')

    info('*** Starting DHCP server on dhcp at 10.0.0.50\n')
    dhcp = net.get('dhcp')
    dhcp.cmd('dhcpd -f -cf /etc/dhcp/dhcpd.conf -pf /var/run/dhcpd.pid')

    info('*** {0} waiting for IP address...\n'.format(h1.name))
    h1.cmd('dhclient')

    info('*** {0} is now using nameserver 8.8.8.8\n'.format(h1.name))

    # Start serving benign content initially
    benign_content = "<html><body><h1>Welcome to Illinois State University</h1></body></html>"
    benign_server_thread = threading.Thread(target=start_web_server, args=(80, benign_content))
    benign_server_thread.start()

    # Wait for a while before launching the attack
    time.sleep(5)

    # Start serving malicious content
    malicious_content = "<html><body><h1>Your computer has been hacked!</h1></body></html>"
    evil_server_thread = threading.Thread(target=start_web_server, args=(80, malicious_content))
    evil_server_thread.start()

    # Start CLI
    CLI(net)

    # Stop Mininet
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    info('*** Creating network\n')
    dhcp_attack()
