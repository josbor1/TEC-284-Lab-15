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
from mininet.term import makeTerms

# Import for DHCP server functions and data (unchanged)

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
    """
    def __init__( self, *args, **kwargs ):
        Topo.__init__( self, *args, **kwargs )
        # Adjust IP addresses to fit 192.168.1.X network
        client = self.addHost( 'h1', ip='0.0.0.0/0' )  # Let DHCP assign IP
        switch = self.addSwitch( 's1' )
        dhcp = self.addHost( 'dhcp', ip='192.168.1.50/24' )
        evil = self.addHost( 'evil', ip='192.168.1.66/24' )

        self.addLink( client, switch )
        self.addLink( evil, switch )
        self.addLink( dhcp, switch, bw=10, delay='500ms' )

# DHCP server functions and data (unchanged)

def make_dhcp_config( filename, intf, gw, dns ):
    "Create a DHCP configuration file"
    config = (
        'interface %s' % intf,
        DNSTemplate,
        'option router %s' % gw,
        'option dns %s' % dns,
        '' )
    with open( filename, 'w') as f:
        f.write( '\n'.join( config ) )

def start_dhcp_server( topo, gw, dns ):  # Accepts topo object or evil host (depending on usage)
    "Start DHCP server on host with specified DNS server"
    if hasattr(topo, 'evil'):  # Check if topo object has evil attribute (called from DHCPTopo)
        evil = topo.evil  # Access evil from topo object
    else:
        evil = topo  # Assume topo is the evil host object (called from run_demo)
    info( '* Starting DHCP server on', evil, 'at', evil.IP(), '\n' )
    dhcp_config = '/tmp/%s-udhcpd.conf' % evil
    make_dhcp_config( dhcp_config, evil.defaultIntf(), gw, dns )
    evil.cmd( 'udhcpd -f', dhcp_config,
              '1>/tmp/%s-dhcp.log 2>&1  &' % evil )

def stop_dhcp_server( host ):
     # (unchanged)

def hacked_webpage(path):
    """Creates a simple HTML file that displays a 'hacked' message"""
    content = """<html><body><h1>YOU HAVE BEEN HACKED!!!</h1></body></html>"""
    with open(path, 'w') as f:
        f.write(content)

def run_demo():
    "Create and run the test network"
    setLogLevel( 'info' )  # Set logging level

    # Check for required executables
    check_required()

    # Create our network
    net = Mininet( topo=DHCPTopo(), link=TCLink, autoStaticArp=True )

    # Start the network
    net.start()

    # User Interaction
    print( "Network started. h1 (victim) has no internet yet." )
    print( "h1 can access the internet before the attack (e.g., try pinging 8.8.8.8)" )
    input("Press Enter to simulate DHCP request from h1 and launch the attack...")

    # Start DHCP servers (legitimate and evil)
    start_dhcp_server( net, '192.168.1.1', '8.8.8.8' )  # Legitimate DHCP with external DNS
    start_dhcp_server( net.evil, '192.168.1.1', '192.168.1.66' )  # Evil DHCP with evil DNS

    print( "h1 should now have internet access (evil DNS server in control)." )
    print( "h1 will be redirected to a 'hacked' webpage when trying to browse (e.g., try curl amazon.com)" )
    input("Press Enter to stop the network...")

    # Stop DHCP servers and clean up
    stop_dhcp_server( net.dhcp )
    stop_dhcp_server( net.evil )
    net.evil.cmd('rm -rf /var/www/html/evil')  # Remove hacked webpage directory
    net.evil.cmd('service dnsmasq stop')  # Stop dnsmasq
    net.stop()

if __name__ == '__main__':
    run_demo()

    
