#!/usr/bin/python3

"""
Rogue DHCP server demo for Stanford CS144 with NAT integration using mn --nat, modified for Raspberry Pi network (192.168.1.X).

This scenario simulates a network where a rogue DHCP server (evil)
provides malicious information to a victim host (h1). The evil host also
acts as a rogue DNS server, redirecting traffic to a pre-defined webpage
after the attack.

We utilize the `mn --nat` flag for NAT functionality.
"""

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.util import quietRun
from mininet.log import setLogLevel, info

class DHCPTopo(Topo):
    """Topology for DHCP Demo:
       h1 - switch - evil (rogue)
                     |
                   dhcp (legitimate)
    """
    def build(self, *args, **kwargs):
        # Adjust IP addresses to fit 192.168.1.X network
        client = self.addHost('h1', ip='0.0.0.0/0')  # Let DHCP assign IP
        switch = self.addSwitch('s1')
        dhcp = self.addHost('dhcp', ip='192.168.1.50/24')
        evil = self.addHost('evil', ip='192.168.1.66/24')

        self.addLink(client, switch)
        self.addLink(evil, switch)
        self.addLink(dhcp, switch, bw=10, delay='500ms')

def start_dhcp_server(host, gw, dns):
    "Start DHCP server on host with specified DNS server"
    info('* Starting DHCP server on', host, 'at', host.IP(), '\n')
    dhcp_config = '/tmp/%s-udhcpd.conf' % host
    make_dhcp_config(dhcp_config, host.defaultIntf(), gw, dns)
    host.cmd('udhcpd -f', dhcp_config,
             '1>/tmp/%s-dhcp.log 2>&1  &' % host)

def stop_dhcp_server(host):
    "Stop DHCP server on host"
    info('Stopping DHCP server on', host)
    host.cmd('pkill -f udhcpd')

def run_demo():
    "Create and run the test network"

    setLogLevel('info')  # Set logging level

    # Create our network
    net = Mininet(topo=DHCPTopo(), link=TCLink, autoStaticArp=True)

    # Start the network
    net.start()

    # User Interaction
    print("Network started. h1 (victim) has no internet yet.")
    print("h1 can access the internet before the attack (e.g., try pinging 8.8.8.8)")
    input("Press Enter to simulate DHCP request from h1 and launch the attack...")

    # Start DHCP servers (legitimate and evil)
    start_dhcp_server(net.get('dhcp'), '192.168.1.1', '8.8.8.8')  # Legitimate DHCP with external DNS
    start_dhcp_server(net.get('evil'), '192.168.1.1', '192.168.1.66')  # Evil DHCP with evil DNS

    print("h1 should now have internet access (evil DNS server in control).")
    print("h1 will be redirected to a 'hacked' webpage when trying to browse (e.g., try curl amazon.com)")
    input("Press Enter to stop the network...")

    # Stop DHCP servers and clean up
    stop_dhcp_server(net.get('dhcp'))
    stop_dhcp_server(net.get('evil'))
    net.get('evil').cmd('rm -rf /var/www/html/evil')  # Remove hacked webpage directory
    net.get('evil').cmd('service dnsmasq stop')  # Stop dnsmasq
    net.stop()

if __name__ == '__main__':
    run_demo()
