from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.util import quietRun
from mininet.log import setLogLevel, info
from mininet.term import makeTerms
from sys import exit, stdin
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

def mountPrivateResolvconf(host):
    "Create/mount private /etc/resolv.conf for host"
    etc = '/tmp/etc-%s' % host
    host.cmd('mkdir -p', etc)
    host.cmd('mount --bind /etc', etc)
    host.cmd('mount -n -t tmpfs tmpfs /etc')
    host.cmd('ln -s %s/* /etc/' % etc)
    host.cmd('rm /etc/resolv.conf')
    host.cmd('cp %s/resolv.conf /etc/' % etc)

def unmountPrivateResolvconf(host):
    "Unmount private /etc dir for host"
    etc = '/tmp/etc-%s' % host
    host.cmd('umount /etc')
    host.cmd('umount', etc)
    host.cmd('rmdir', etc)

def dhcpdemo(firefox=True):
    "Rogue DHCP server demonstration"
    checkRequired()
    topo = DHCPTopo()
    net = Mininet(topo=topo, link=TCLink)
    h1, dhcp, evil = net.get('h1', 'dhcp', 'evil')
    mountPrivateResolvconf(h1)
    # Set up a good but slow DHCP server
    startDHCPserver(dhcp, gw='10.0.0.1', dns='8.8.8.8')
    startDHCPclient(h1)
    waitForIP(h1)
    # Make sure we can fetch the good google.com
    info('* Fetching google.com:\n')
    print(h1.cmd('curl google.com'))
    # For firefox, start it up and tell user what to do
    if firefox:
        net.terms += makeTerms([h1], 'h1')
        h1.cmd('firefox www.stanford.edu -geometry 400x400-50+50 &')
        print('*** You may want to do some DNS lookups using dig')
        print('*** Please go to amazon.com in Firefox')
        print('*** You may also wish to start up wireshark and look at bootp and/or dns')
        prompt("*** Press return to start up evil DHCP server: ")
    # Now start up an evil but fast DHCP server
    startDHCPserver(evil, gw='10.0.0.1', dns=evil.IP())
    # And an evil fake DNS server
    startFakeDNS(evil)
    # And an evil web server
    startEvilWebServer(evil)
    h1.cmd('ifconfig', h1.defaultIntf(), '0')
    waitForIP(h1)
    info('* New DNS result:\n')
    info(h1.cmd('host google.com'))
    # Test http request
    if firefox:
        print("*** You may wish to look at DHCP and DNS results in wireshark")
        print("*** You may also wish to do some DNS lookups using dig")
        print("*** Please go to google.com in Firefox")
        print("*** You may also want to try going back to amazon.com and hitting shift-refresh")
    else:
        info('* Fetching google.com:\n')
        print(h1.cmd('curl google.com'))
    if firefox:
        prompt("*** Press return to shut down evil DHCP/DNS/Web servers: ")
    # Clean up everything
    stopFakeDNS(evil)
    stopEvilWebServer(evil)
    stopDHCPserver(evil)
    if firefox:
        print("*** Try going to some other web sites if you like")
        prompt("*** Press return to exit: ")
    stopDHCPserver(dhcp)
    stopDHCPclient(h1)
    net.stop()

# Run the dhcpdemo function when the script is executed
if __name__ == '__main__':
    setLogLevel('info')
    dhcpdemo()
