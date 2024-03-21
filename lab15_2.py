from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, Host
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def create_network():
    net = Mininet(controller=Controller, switch=OVSKernelSwitch, link=TCLink)

    # Add controller
    net.addController('c0')

    # Add hosts
    h1 = net.addHost('h1', ip='10.0.0.10/24')
    dhcp = net.addHost('dhcp', ip='10.0.0.50/24')
    evil = net.addHost('evil', ip='10.0.0.66/24')

    # Add switch
    s1 = net.addSwitch('s1')

    # Add links
    net.addLink(h1, s1)
    net.addLink(dhcp, s1, bw=10, delay='500ms')
    net.addLink(evil, s1)

    # Add NAT connectivity
    nat = net.addNAT()
    net.addLink(nat, s1)

    return net

def start_dhcp_server(host, gw):
    host.cmd('dnsmasq -k -A /#/%s 1>/tmp/dns.log 2>&1 &' % host.IP())
    host.cmd('udhcpd -f /etc/udhcpd.conf &')

def stop_dhcp_server(host):
    host.cmd('kill %dnsmasq')
    host.cmd('kill %udhcpd')

def start_evil_web_server(host):
    host.cmd('python -m SimpleHTTPServer 80 >& /tmp/http.log &')

def stop_evil_web_server(host):
    host.cmd('kill %python')

def main():
    setLogLevel('info')
    net = create_network()
    net.start()

    h1, dhcp, evil = net.get('h1', 'dhcp', 'evil')

    start_dhcp_server(dhcp, gw='10.0.0.1')
    start_evil_web_server(evil)

    # Wait for h1 to obtain IP address
    info('* Waiting for h1 to obtain IP address...')
    h1.cmd('dhclient h1-eth0')
    info('done.\n')

    # Display network information
    info('* Network configuration:\n')
    info(h1.cmd('ifconfig'))
    info(dhcp.cmd('ifconfig'))
    info(evil.cmd('ifconfig'))

    # Display DNS configuration
    info('\n* DNS configuration:\n')
    info(h1.cmd('cat /etc/resolv.conf'))

    # Fetch a website to demonstrate connectivity
    info('\n* Fetching google.com from h1:\n')
    info(h1.cmd('curl google.com'))

    # Start CLI
    CLI(net)

    # Clean up
    stop_evil_web_server(evil)
    stop_dhcp_server(dhcp)
    net.stop()

if __name__ == '__main__':
    main()
