FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
       python3-venv git curl openssh-server  \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ARG USERNAME
ARG PASSWORD

RUN useradd -m -s /bin/bash "${USERNAME}" && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd && \
    install -d -o "${USERNAME}" -g "${USERNAME}" /home/"${USERNAME}"/applied-ai-workbench

WORKDIR /home/${USERNAME}/applied-ai-workbench

RUN mkdir -p /var/run/sshd && \
    ssh-keygen -A && \
    sed -i 's/^#\\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config && \
    echo "AllowUsers ${USERNAME}" >> /etc/ssh/sshd_config

CMD ["/usr/sbin/sshd", "-D"]
