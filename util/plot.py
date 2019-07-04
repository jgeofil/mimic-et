from matplotlib import pyplot as plt

def con_avg(l):
	n = 5
	return [sum(l[i:i+n])/n for i in range(0, len(l), n)]

def plot_gan(d_loss, g_loss, d_real, d_fake_pre, d_fake_post, out_folder):
	plt.close()
	fig, (ax1, ax2) = plt.subplots(2)
	ax1.plot(con_avg(d_loss), label='Loss D')
	ax1.plot(con_avg(g_loss), label='Loss G')
	ax1.grid()
	ax1.legend()
	ax2.plot(con_avg(d_real), label='D(x)')
	ax2.plot(con_avg(d_fake_pre), label='D(G(z))')
	ax2.plot(con_avg(d_fake_post), label='New D(G(z))')
	ax2.grid()
	ax2.legend()
	plt.savefig(out_folder+'/loss.png')